#include <ode/odeconfig.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <unistd.h>

#include "neat.h"
#include "organism.h"
#include "noveltyset.h"
#include "datarec.h"
#include <ode/ode.h>
#include "biped.h"
#include "experiments.h"
#include "calc_evol.h"
#include "graph.h"

static vector<float> best_fits;
static plot front_plot;
static plot fitness_plot;

extern bool evaluate_switch;
extern bool seed_mode;
extern bool minimal_criteria;
extern bool population_dirty;
extern char output_dir[30];
extern char seed_name[40];

dReal MAXTORQUE_KNEE= 5.0;
dReal MAXTORQUE_HIPMINOR= 5.0;
dReal MAXTORQUE_HIPMAJOR= 5.0;
dReal P_CONSTANT=9.0;
dReal D_CONSTANT=0.0;

static ofstream *logfile;

int novelty_function = NF_COGSAMPSQ;
vector<dGeomID> geoms;
vector<Creature*> creatures;

void initialize_biped(const char* nefile) {
  NEAT::load_neat_params(nefile,true);
}

//NEAT + NS stuff
inline float dist(float x1, float y1, float x2, float y2)
{
  float xd = x1-x2;
  float yd = y1-y2;
  return xd*xd+yd*yd;
}

static	void calculate_delta(dVector3 v1, dVector3 v2, dVector3 o)
{
  for (int x=0; x<3; x++)
    o[x]=v2[x]-v1[x];
}

static	void calculate_power(dVector3 v,int pow)
{
  for (int x=0; x<3; x++)
  {
    float temp=v[x];
    bool sign=false;
    if (temp<0.0)
      sign=true;
    for (int k=1; k<pow; k++)
      v[x]*=temp;
    if (sign)
      v[x]=(-v[x]);
  }
}

//novelty metric for maze simulation
float walker_novelty_metric(noveltyitem* x,noveltyitem* y)
{
  float dist=0.0;

  int size = x->data[0].size();
  int size2 = y->data[0].size();
  if (size!=size2) {
    cout << size << " " << size2 << endl;
    exit(0);
  }

  for (int k=0; k<size; k++)
  {
    float delta = x->data[0][k]-y->data[0][k];
    dist+=delta*delta;
  }

  return dist;

}

static dWorldID world;
static dSpaceID space;
static dGeomID floorplane;

static vector<dBodyID> bodies;

static dJointGroupID contactgroup;

// this is called by dSpaceCollide when two objects in space are
// potentially colliding.
static void nearCallback (void *data, dGeomID o1, dGeomID o2)
{
  dBodyID b1,b2;
  dBodyID test;
  assert(o1);
  assert(o2);

  b1 = dGeomGetBody(o1);
  b2 = dGeomGetBody(o2);

  if (b1 && b2 && dAreConnected (b1,b2)) return;

  if (o1 == floorplane || o2 == floorplane)
  {
    if (o1==floorplane)
      test=b2;
    if (o2==floorplane)
      test=b1;
    //test should equal the body that is colliding with floor

    for (int x=0; x<creatures.size(); x++)
    {
      int bsize=creatures[x]->bodies.size();
      for (int y=0; y<bsize; y++)
        if (test==creatures[x]->bodies[y])
          creatures[x]->onground[y]=true;
    }
  }

  const int N = 32;
  dContact contact[N];
  int n = dCollide (o1,o2,N,&(contact[0].geom),sizeof(dContact));
  if (n > 0)
  {
    for (int i=0; i<n; i++)
    {
      contact[i].surface.mode = 0;
      contact[i].surface.mu = dInfinity; //50.0; // was: dInfinity
      dJointID c = dJointCreateContact (world,contactgroup,&contact[i]);
      dJointAttach (c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
    }
  }
}

// Executes one step of simulation
void simulationStep(const bool bMoviePlay)
{
  const double timestep = 0.01;
  if(!bMoviePlay)
  {
    dSpaceCollide(space, 0, &nearCallback);
    dWorldStep(world, timestep);
  }

  for(int x = 0; x < creatures.size(); ++x)
    creatures[x]->Update(timestep);

  if(!bMoviePlay)
    dJointGroupEmpty(contactgroup);
}

void create_world(Controller *controller, bool log, bool bMoviePlay)
{
  // Create world
  dRandSetSeed(10);
  dInitODE();
  creatures.clear();
  world = dWorldCreate();
  space = dHashSpaceCreate(0);
  contactgroup = dJointGroupCreate(0);
     
  // Grab gravity settings from world
  CTRNNController *cc= (CTRNNController*)controller;
  const float discrete = ((1.0 - cc->genes->traits[0]->params[4]) * 4.0) + 1.0; 
  float gravitysetting = -9.8 * (discrete / 5.0);
  const float discrete2 = ((1.0 - cc->genes->traits[0]->params[5]) * 4.0) + 1.0; 
  //cout << "gravity: " << gravitysetting << endl;
  gravitysetting=-9.8;
  P_CONSTANT = 9;

  /*
  gravitysetting= -9.8 * NEAT::gravity;
  //P_CONSTANT = 9 ; //4.5 + 4.5 * NEAT::gravity;// * (discrete/3);
  P_CONSTANT = 4.5 + 4.5 * NEAT::gravity;// * (discrete/3);
  MAXTORQUE_KNEE= 2.5+2.5* NEAT::gravity; //5.0;
  MAXTORQUE_HIPMINOR= 2.5+2.5*NEAT::gravity; //5.0;
  MAXTORQUE_HIPMAJOR= 2.5+2.5*NEAT::gravity; //5.0;
  */
  
  dWorldSetGravity(world, 0, 0, gravitysetting);
    
  floorplane = dCreatePlane(space, 0, 0, 1, 0.0);
  dWorldSetERP(world, 0.1);
  dWorldSetCFM(world, 1E-4);

  Biped *biped = new Biped(log, bMoviePlay);
  dVector3 pos = {0.0, 0.0, 0.0};

  biped->Create(world, space, pos, controller);
  creatures.push_back(biped);
}

void destroy_world()
{
  dJointGroupEmpty(contactgroup);
  dJointGroupDestroy(contactgroup);

  for(int x = 0; x < geoms.size(); ++x)
    dGeomDestroy(geoms[x]);

  for(int x = 0; x < creatures.size(); ++x)
  {
    creatures[x]->Destroy();
    delete creatures[x];
  }
  
  creatures.clear();
  bodies.clear();
  geoms.clear();

  dSpaceDestroy(space);
  dWorldDestroy(world);
  dCloseODE();
}

void update_behavior(vector<float> &k, Creature* c,bool good=true,float time=0.0)
{
   
  if (novelty_function==NF_COGSAMPSQ)
  {
    dVector3& o_com= ((Biped*)c)->orig_com;
    dVector3& c_com= ((Biped*)c)->curr_com;
    dVector3 com;
    dVector3 delta;
    c->CenterOfMass(com);
    calculate_delta(o_com,com,delta);
    //calculate_power(delta,2);

    if (good)
    {
      k.push_back(delta[0]);
      k.push_back(delta[1]);
    }
    else
    {
      k.push_back(0.0);
      k.push_back(0.0);
    }
  }
  if(novelty_function==NF_TRAIT) {

    Biped* b = ((Biped*)c);
    NEAT::Genome* g = ((CTRNNController*)b->controller)->genes;
    k.push_back(g->traits[0]->params[4]); 
    float count = b->lft.size();
    if(b->rft.size()<count) count=b->rft.size();
    count/=7.0;
    if(count>1.0) count=1.0;
    //k.push_back(count);
    //k.push_back(time);
    //k.push_back(g->traits[0]->params[1]); 
  }
}

dReal evaluate_controller(Controller* controller, noveltyitem* ni,  data_record *record, bool log)
{
  // Create the ODE world
  create_world(controller, log);
  
  // Simulate until robot falls (or 15 seconds passes)
  vector<float> k;
  int timestep = 0;
  const int simtime = 1500;
  while(!creatures[0]->abort() && timestep < simtime)
  {
    simulationStep();
    ++timestep;
    if(timestep % 100 == 0 && novelty_function % 2 == 1)
    {
      update_behavior(k, creatures[0]);
    }
    
    if(log && timestep % 100 == 0)
      cout << creatures[0]->fitness() << endl;
  }
  const dReal fitness = creatures[0]->fitness();
  if(creatures[0]->abort())
    cout << "CREATURE DIED WITH FITNESS " << fitness << endl;
  else
    cout << "CREATURE TIMED OUT WITH FITNESS " << fitness << endl;
  
  const int time = timestep;

  //for (int x=timestep+1; x<=simtime; x++)
  //    if (x%100==0)
  if(novelty_function % 2 == 1)
  {
    while(k.size() < (simtime/100 * 2))
      update_behavior(k, creatures[0]);
  }
  else
  {
    update_behavior(k, creatures[0], true, (float)time / (float)simtime);
  }

  ((Biped*)creatures[0])->lft.push_back(timestep);
  ((Biped*)creatures[0])->rft.push_back(timestep);
  if(ni != NULL)
  {
    //ni->time=time;
    ni->novelty_scale = 1.0;
    ni->data.push_back(k);
  }

  if(record != NULL)
  {
    dVector3 com;
    creatures[0]->CenterOfMass(com);
    record->ToRec[0] = fitness;
    record->ToRec[1] = com[0];
    record->ToRec[2] = com[1];
    record->ToRec[3] = com[2];
    record->ToRec[4] = timestep;
  }

  destroy_world();
  
  return fitness;
}

noveltyitem* biped_evaluate(NEAT::Organism *org, data_record *data)
{
  noveltyitem *new_item = new noveltyitem;
  new_item->genotype = new Genome(*org->gnome);
  new_item->phenotype = new Network(*org->net);

  CTRNNController *cont = new CTRNNController(org->net, org->gnome);
  new_item->fitness = evaluate_controller(cont, new_item, data, true);
  org->fitness = new_item->fitness;
  new_item->secondary = 0;
  //if (new_item->fitness < 2.5) new_item->viable=false;
  //else new_item->viable=true;
  new_item->viable = true;
  if(get_age_objective())	
    new_item->secondary = -org->age; //time; //fitness;
  else
    new_item->secondary = 0; //time; //fitness;

  delete cont;

  return new_item;
}


//novelty maze navigation run
Population *biped_novelty_realtime(char* outputdir,int par,const char* genes,bool novelty) {

  Population *pop;
  Genome *start_genome;
  char curword[20];

  int id;

  if (outputdir!=NULL) strcpy(output_dir,outputdir);

  if (!seed_mode)
    strcpy(seed_name,genes);
  //starter genes file
  ifstream iFile(seed_name,ios::in);

  cout<<"START BIPED NAVIGATOR NOVELTY REAL-TIME EVOLUTION VALIDATION"<<endl;
  if (!seed_mode)
  {
    cout<<"Reading in the start genome"<<endl;
  }
  else
    cout<<"Reading in the seed genome" <<endl;

  //Read in the start Genome
  iFile>>curword;
  iFile>>id;
  cout<<"Reading in Genome id "<<id<<endl;
  start_genome=new Genome(id,iFile);
  iFile.close();

  cout<<"Start Genome: "<<start_genome<<endl;

  //Spawn the Population from starter gene
  cout<<"Spawning Population off Genome"<<endl;
  if (!seed_mode) 
    pop=new Population(start_genome,NEAT::pop_size);
  else
  {
    pop=new Population(seed_name);//start_genome,NEAT::pop_size,0.0);
    if (evaluate_switch) {
      int dist=0;
      double evol=0.0;
      cout << "Evaluating..." << endl; 
      cout << pop->organisms.size() << endl;
      //evolvability_biped(pop->organisms[0],"dummyfile",&dist,&evol,true);
      //cout << endl << dist << " " << evol << endl;
      noveltyitem* i= biped_evaluate(pop->organisms[0]); 
      cout << "fitness: " << i->fitness << endl;
      return 0;
    }

  }
  cout<<"Verifying Spawned Pop"<<endl;
  pop->verify();
  pop->set_evaluator(&biped_evaluate);
  //pop->set_compatibility(&behavioral_compatibility);
  //Start the evolution loop using rtNEAT method calls
  biped_novelty_realtime_loop(pop,novelty);

  //clean up
  return pop;
}

//actual rtNEAT loop for novelty maze navigation runs
int biped_novelty_realtime_loop(Population *pop,bool novelty) {
  vector<Organism*>::iterator curorg;
  vector<Species*>::iterator curspecies;
  vector<Species*>::iterator curspec; //used in printing out debug info

  vector<Species*> sorted_species;  //Species sorted by max fit org in Species

  //was 1.0*number_of_samples+1.0 for earlier results...
  float archive_thresh=(1.0);// * 20.0 * envList.size(); //initial novelty threshold
  //if(!minimal_criteria)
  //	archive_thresh*=20;
  //if(constraint_switch)
  //archive_thresh/=200.0;
  cout << "Archive threshold: " << archive_thresh << endl;
  //archive of novel behaviors
  noveltyarchive archive(archive_thresh,*walker_novelty_metric,true,30,minimal_criteria);

  data_rec Record; //stores run information

  int count=0;
  int pause;

  //Real-time evolution variables
  int offspring_count;
  Organism *new_org;

  //We try to keep the number of species constant at this number
  int num_species_target=NEAT::pop_size/20;

  //This is where we determine the frequency of compatibility threshold adjustment
  int compat_adjust_frequency = NEAT::pop_size/20;
  if (compat_adjust_frequency < 1)
    compat_adjust_frequency = 1;

  //Initially, we evaluate the whole population
  //Evaluate each organism on a test
  int indiv_counter=0;
  pop->evaluate_all();

  if (novelty) {
    //assign fitness scores based on novelty
    archive.evaluate_population(pop,true);
    //add to archive
    archive.evaluate_population(pop,false);
  }

  if (novelty && minimal_criteria)
    for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg)
  {
    (*curorg)->fitness = SNUM/1000.0;
  }
  //Get ready for real-time loop
  //Rank all the organisms from best to worst in each species
  pop->rank_within_species();

  //Assign each species an average fitness
  //This average must be kept up-to-date by rtNEAT in order to select species probabailistically for reproduction
  pop->estimate_all_averages();

  cout <<"Entering real time loop..." << endl;

  //Now create offspring one at a time, testing each offspring,
  // and replacing the worst with the new offspring if its better
  for(offspring_count=0; offspring_count<NEAT::pop_size*2001; offspring_count++)
  {
    //fix compat_threshold, so no speciation...
    //NEAT::compat_threshold = 1000000.0;
    //only continue past generation 1000 if not yet solved
    //if(offspring_count>=pop_size*1000 && firstflag)
    // if(firstflag)
    // break;

    int evolveupdate=50000;
    if (NEAT::evolvabilitytest && offspring_count % evolveupdate ==0) {
      char fn[100];
      sprintf(fn,"%s_evolvability%d.dat",output_dir,offspring_count/evolveupdate);
      for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {
        evolvability_biped(*curorg,fn);
      }
    }

    //end of generation
    if (offspring_count % (NEAT::pop_size*1) == 0)
    {
      /*
      if((offspring_count/NEAT::pop_size)%change_extinction_length==0)
      change_extinction_point();
      if((offspring_count/NEAT::pop_size)%change_goal_length==0)
      change_goal_location();
      */
      if (population_dirty) {
        pop->evaluate_all();
        population_dirty=false;
      }
      if (novelty) {
        archive.end_of_gen_steady(pop);
        //archive.add_randomly(pop);
        archive.evaluate_population(pop,false);
        cout << "ARCHIVE SIZE:" <<
          archive.get_set_size() << endl;
      }
      double mx=0.0;  
      double tot=0.0;    
      Organism* b;  
      for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {
        tot+=(*curorg)->noveltypoint->fitness;
        if( (*curorg)->noveltypoint->fitness > mx) {
          mx=(*curorg)->noveltypoint->fitness; b=(*curorg);
        }
      } 
      cout << "GENERATION " << offspring_count/NEAT::pop_size << ": total = " << tot << ", max = " << mx <<  endl;

      //evolvability_biped(b,"dummy");
      char fn[100];
      sprintf(fn,"%sdist%d",output_dir,offspring_count/NEAT::pop_size);
      if (NEAT::printdist)
        pop->print_distribution(fn);
    }

    //write out current generation and fittest individuals
    if ( offspring_count % (NEAT::pop_size*NEAT::print_every) == 0 )
    {
      cout << offspring_count << endl;
      char fname[100];
      sprintf(fname,"%sarchive.dat",output_dir);
      archive.Serialize(fname);

      sprintf(fname,"%sfittest_%d",output_dir,offspring_count/NEAT::pop_size);
      archive.serialize_fittest(fname);

      sprintf(fname,"%sgen_%d",output_dir,offspring_count/NEAT::pop_size);
      pop->print_to_file_by_species(fname);


      sprintf(fname,"%srecord.dat",output_dir);
      Record.serialize(fname);
    }

    //Every pop_size reproductions, adjust the compat_thresh to better match the num_species_targer
    //and reassign the population to new species
    if (offspring_count % compat_adjust_frequency == 0) {
      count++;
#ifdef DEBUG_OUTPUT
      cout << "Adjusting..." << endl;
#endif
      if (novelty) {
        //update fittest individual list
        archive.update_fittest(pop);
        //refresh generation's novelty scores
        archive.evaluate_population(pop,true);
      }
      int num_species = pop->species.size();
      double compat_mod=0.1;  //Modify compat thresh to control speciation
      // This tinkers with the compatibility threshold
      if (num_species < num_species_target) {
        NEAT::compat_threshold -= compat_mod;
      }
      else if (num_species > num_species_target)
        NEAT::compat_threshold += compat_mod;

      if (NEAT::compat_threshold < 0.3)
        NEAT::compat_threshold = 0.3;
#ifdef DEBUG_OUTPUT
      cout<<"compat_thresh = "<<NEAT::compat_threshold<<endl;
#endif

      //Go through entire population, reassigning organisms to new species
      for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {
        pop->reassign_species(*curorg);
      }
    }


    //For printing only
#ifdef DEBUG_OUTPUT
    for (curspec=(pop->species).begin(); curspec!=(pop->species).end(); curspec++) {
      cout<<"Species "<<(*curspec)->id<<" size"<<(*curspec)->organisms.size()<<" average= "<<(*curspec)->average_est<<endl;
    }

    cout<<"Pop size: "<<pop->organisms.size()<<endl;
#endif

    //Here we call two rtNEAT calls:
    //1) choose_parent_species() decides which species should produce the next offspring
    //2) reproduce_one(...) creates a single offspring fromt the chosen species
    new_org=(pop->choose_parent_species())->reproduce_one(offspring_count,pop,pop->species);

    //Now we evaluate the new individual
    //Note that in a true real-time simulation, evaluation would be happening to all individuals at all times.
    //That is, this call would not appear here in a true online simulation.
#ifdef DEBUG_OUTPUT
    cout<<"Evaluating new baby: "<<endl;
#endif

    /*	data_record* newrec=new data_record();
    newrec->indiv_number=indiv_counter;
    //evaluate individual, get novelty point
    new_org->noveltypoint = maze_novelty_map(new_org,newrec);
    new_org->noveltypoint->indiv_number = indiv_counter;
    new_org->fitness=new_org->noveltypoint->fitness;
    */
    data_record* newrec=new_org->datarec;
    //calculate novelty of new individual
    if (novelty) {
      archive.evaluate_individual(new_org,pop->organisms);
      //production of novelty tracking...
      int looking_for = new_org->gnome->parent_id;
      for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {
        if((*curorg)->gnome->genome_id==looking_for) {
          (*curorg)->gnome->production+=new_org->noveltypoint->novelty;
          (*curorg)->gnome->production_count++;
          //cout << "Parent " << looking_for << " found...parent avg prod: " << (*curorg)->gnome->production/(*curorg)->gnome->production_count << endl;
        }
      }
        
      //newrec->ToRec[5] = archive.get_threshold();
      newrec->ToRec[6] = archive.get_set_size();
      newrec->ToRec[RECSIZE-2] = new_org->noveltypoint->novelty;
    }
    if ( !new_org->noveltypoint->viable && minimal_criteria)
    {
      new_org->fitness = SNUM/1000.0;
      //new_org->novelty = 0.00000001;
      //reset behavioral characterization
      new_org->noveltypoint->reset_behavior();
      cout << "fail" << endl;
      //  cout << " :( " << endl;
    }
    else
    {
      // cout << ":)" << new_org->noveltypoint->indiv_number << endl;
    }
    //add record of new indivdual to storage
    //Record.add_new(newrec);
    indiv_counter++;

    //update fittest list
    archive.update_fittest(new_org);
#ifdef DEBUG_OUTPUT
    cout << "Fitness: " << new_org->fitness << endl;
    cout << "Novelty: " << new_org->noveltypoint->novelty << endl;
    cout << "RFit: " << new_org->noveltypoint->fitness << endl;
#endif

    //Now we reestimate the baby's species' fitness
    new_org->species->estimate_average();

    //Remove the worst organism
    //if(rand_repl || fitness_measure ==fitness_rnd)
    // pop->remove_random();
    //else
        
    pop->remove_worst();
    /*
    if(randfloat()<0.99)
    pop->remove_worst();
    else 
    pop->remove_old();
    */
  }

  //write out run information, archive, and final generation
  cout << "COMPLETED...";
  char filename[100];
  sprintf(filename,"%srecord.dat",output_dir);
  char fname[100];
  sprintf(fname,"%srtarchive.dat",output_dir);
  archive.Serialize(fname);
  //Record.serialize(filename);

  sprintf(fname,"%sfittest_final",output_dir);
  archive.serialize_fittest(fname);

  sprintf(fname,"%srtgen_final",output_dir);
  pop->print_to_file_by_species(fname);
  delete pop;
  exit(0);
  return 0;
}

#define BIPEDMUTATIONS 200
#define BDIM 30
void evolvability_biped(Organism* org,char* fn,int* di,double* ev,bool recall) {
  fstream file;
  file.open(fn,ios::app|ios::out);
  cout <<"Evolvability..." << endl;
  // file << "---" <<  " " << org->winner << endl;
  double points[BIPEDMUTATIONS*BDIM];
  float minx=-10.0,maxx=10.0,miny=-10.0,maxy=10.0;
  double ox,oy,fit;
  int nodes;
  int connections;
  data_record rec;
  for (int i=0; i<BIPEDMUTATIONS; i++) {
    Genome *new_gene= new Genome(*org->gnome);
    //new_org->gnome = new Genome(*org->gnome);
        
    if (i!=0) //first copy is clean
      for (int j=0; j<1; j++) mutate_genome(new_gene);
    Organism *new_org= new Organism(0.0,new_gene,0);

    noveltyitem* nov_item = biped_evaluate(new_org,&rec);
    if (i==0) {
      fit=nov_item->fitness;
      nodes=new_org->net->nodecount();
      connections=new_org->net->linkcount();
      ox=rec.ToRec[1];
      oy=rec.ToRec[2];
    }
    if(recall)
    {
      for(int k=0;k<nov_item->data[0].size();k++)
        file << nov_item->data[0][k] << " ";
      file << endl; 
    }   
    //file << rec.ToRec[1] << " " << rec.ToRec[2]<< endl;
    for(int k=0;k<nov_item->data[0].size();k++) {
      points[i*BDIM+k]=nov_item->data[0][k]/25.0;
    }
    /*   
    points[i*2]=(rec.ToRec[1]-minx)/(maxx-minx);
    points[i*2+1]=(rec.ToRec[2]-miny)/(maxy-miny);
    cout << points[i*2] << " " << points[i*2+1] << endl;
    */
    delete new_org;
    delete nov_item;
    //file << endl;
  }
  int dist = distinct(points,BIPEDMUTATIONS,BDIM);
  if (di!=NULL) *di=dist;
  double evol = 0; //test_indiv(points,BIPEDMUTATIONS);
  if (ev!=NULL) *ev=evol;
  if(!recall) {
    file << dist << " " << evol << " " << ox << " " << oy << " " << nodes << " " <<connections << " " << fit << endl;
    file.close();
  }
}

Population *biped_alps(char* output_dir, const char *genes, int gens, bool novelty) {
  population_state* p_state = create_biped_popstate(output_dir,genes,gens,novelty);
    
  alps k(5,20,p_state->pop->start_genome,p_state,biped_success_processing,output_dir);
  k.do_alps();
}

static int maxgens;
static int push_back_size = 20;

Population *biped_generational(char* outputdir,const char *genes, int gens,bool novelty)
{
  char logname[100];
  sprintf(logname,"%s_log.txt",outputdir);
  logfile=new ofstream(logname);

  population_state* p_state = create_biped_popstate(outputdir, genes, gens, novelty);
  for(int gen = 0; gen <= maxgens; gen++)  { //WAS 1000
    cout << "Generation " << gen << endl;
    const bool win = biped_generational_epoch(p_state,gen);
    p_state->pop->epoch(gen);
  }
  delete logfile;
  return p_state->pop;

}

population_state* create_biped_popstate(char* outputdir, const char *genes, int gens, bool novelty) {
  maxgens = gens;
  
  const float archive_thresh = 3.0;
  noveltyarchive *archive = new noveltyarchive(archive_thresh, *walker_novelty_metric, true, push_back_size, minimal_criteria, true);

  // If doing multiobjective, turn off speciation
  // TODO: Maybe turn off elitism
  if (NEAT::multiobjective)
    NEAT::speciation=false;

  data_rec Record;

  if(outputdir != NULL)
    strcpy(output_dir, outputdir);
  
  // Read in the start Genome
  cout << "Reading in the start genome..." << endl;
  char curword[20];
  int id;
  ifstream iFile(genes, ios::in);
  iFile >> curword;
  iFile >> id;
  cout << "Reading in genome with id " << id << "..." << endl;
  Genome *start_genome = new Genome(id, iFile);
  iFile.close();
  cout << "Read genome" << endl;
  
  // Output start genome
  cout<<"Start Genome: " << start_genome << endl;

  // Spawn the Population
  cout << "Spawning population from genome..." << endl;
  Population *pop = new Population(start_genome,NEAT::pop_size);
  cout << "Spawned population" << endl;

  cout << "Verifying spawned population..." << endl;
  pop->verify();
  cout << "Verified spawned population" << endl;

  // Set evaluator and evaluate initial population
  cout << "Evaluating initial population..." << endl;
  pop->set_evaluator(&biped_evaluate);
  pop->evaluate_all();
  cout << "Evaluated initial population" << endl;
  
  return new population_state(pop, novelty, archive);
  //pop->set_compatibility(&behavioral_compatibility);
}

int biped_success_processing(population_state* pstate) {
  double& best_fitness = pstate->best_fitness;
  double& best_secondary = pstate->best_secondary;

  vector<Organism*>::iterator curorg;
  Population* pop = pstate->pop;
  //Evaluate each organism on a test
  int indiv_counter=0;
  for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
    if ((*curorg)->noveltypoint->fitness > best_fitness)
    {
      best_fitness = (*curorg)->noveltypoint->fitness;
      cout << "NEW BEST: " << best_fitness << endl;
      
      //char filename[100];
      //sprintf(filename,"%s_winner", output_dir);
      //(*curorg)->print_to_file(filename);
    }

    indiv_counter++;
    if ((*curorg)->noveltypoint->viable && !pstate->mc_met)
      pstate->mc_met=true;
    else if (pstate->novelty && !(*curorg)->noveltypoint->viable && minimal_criteria && pstate->mc_met)
    {
      destroy_organism((*curorg));
    }

    if (!pstate->novelty)
      (*curorg)->fitness = (*curorg)->noveltypoint->fitness;
  }

  if(logfile!=NULL)
    (*logfile) << pstate->generation*NEAT::pop_size<< " " << best_fitness << " " << best_secondary << endl;
  //(*logfile) << best_fitness << " " << best_secondary << endl;
  return 0;
}

//int biped_generational_epoch(Population **pop2,int generation,data_rec& Record, noveltyarchive& archive, bool novelty) {
int biped_generational_epoch(population_state* p, int gen) {
  generalized_generational_epoch(p,gen,&biped_success_processing); 
}
/*
Population* pop= *pop2;
vector<Organism*>::iterator curorg;
vector<Species*>::iterator curspecies;
static double best_fitness =0.0;
static double best_secondary =  -100000.0;
static vector<Organism*> measure_pop;

static bool win=false;
static bool firstflag=false;
int winnernum;
int indiv_counter=0;

int evolveupdate=100;
if (generation==0) pop->evaluate_all();
bool speciation=false;

if(!speciation)
if (NEAT::multiobjective) {  //merge and filter population
for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
measure_pop.push_back(new Organism(*(*curorg),true)); //TODO:maybe make a copy?
}

//evaluate this 'super-population'
archive.rank(measure_pop);
if (generation!=0) {
//chop population down by half (maybe delete orgs that aren't used)
int start=measure_pop.size()/2;
vector<Organism*>::iterator it;
for (it=measure_pop.begin()+start; it!=measure_pop.end(); it++)
delete (*it);
measure_pop.erase(measure_pop.begin()+(measure_pop.size()/2),measure_pop.end());
}
//delete old pop, create new pop
Genome* sg=pop->start_genome;
delete pop;
pop=new Population(measure_pop);
pop->set_startgenome(sg);
pop->set_evaluator(&biped_evaluate);
*pop2= pop;
}

if (NEAT::evolvabilitytest && generation%evolveupdate==0)
{
char fn[100];
sprintf(fn,"%s_evolvability%d.dat",output_dir,generation/evolveupdate);
for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {
evolvability_biped(*curorg,fn);
}
}

//Evaluate each organism on a test
for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {

//newrec->indiv_number=indiv_counter;
//data_record* newrec=new data_record();
//evaluate individual, get novelty point
//(*curorg)->noveltypoint = maze_novelty_map((*curorg),newrec);
//(*curorg)->noveltypoint->indiv_number = indiv_counter;
//(*curorg)->datarec = newrec;
data_record* newrec = (*curorg)->datarec;

if ((*curorg)->noveltypoint->secondary >best_secondary) {
best_secondary=(*curorg)->noveltypoint->secondary;
cout << "NEW BEST SEC " << best_secondary << endl;

}

if ((*curorg)->noveltypoint->fitness > best_fitness)
{
best_fitness = (*curorg)->noveltypoint->fitness;
cout << "NEW BEST " << best_fitness << endl;
}

//add record of new indivdual to storage
//TODO: PUT BACK IN (to fix record.dat...)
//Record.add_new(newrec);
indiv_counter++;
if ( !(*curorg)->noveltypoint->viable && minimal_criteria)
{
(*curorg)->fitness = SNUM/1000.0;
//new_org->novelty = 0.00000001;
//reset behavioral characterization
(*curorg)->noveltypoint->reset_behavior();
//cout << "fail" << endl;
// cout << " :( " << endl;
}

//update fittest list
archive.update_fittest(*curorg);

if (!novelty)
(*curorg)->fitness = (*curorg)->noveltypoint->fitness;
}

//write line to log file
cout << "writing line to log" << endl;
(*logfile) << generation << " " << best_fitness << " " << best_secondary << endl;

if (novelty)
{

//NEED TO CHANGE THESE TO GENERATIONAL EQUIVALENTS...
//assign fitness scores based on novelty
archive.evaluate_population(pop,true);
///now add to the archive (maybe remove & only add randomly?)
archive.evaluate_population(pop,false);

	
#ifdef PLOT_ON
vector<float> x,y,z;
pop->gather_objectives(&x,&y,&z);
front_plot.plot_data(x,y,"p","Pareto front");
best_fits.push_back(best_fitness);
fitness_plot.plot_data(best_fits,"lines","Fitness");
#endif

if (NEAT::multiobjective)
archive.rank(pop->organisms);

pop->print_divtotal();
for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
if ( !(*curorg)->noveltypoint->viable && minimal_criteria)
{
(*curorg)->fitness = SNUM/1000.0;
//new_org->novelty = 0.00000001;
//reset behavioral characterization
//cout << "fail" << endl;
// cout << " :( " << endl;
}
}
cout << "ARCHIVE SIZE:" << archive.get_set_size() << endl;
cout << "THRESHOLD:" << archive.get_threshold() << endl;
archive.end_of_gen_steady(pop);
//adjust novelty of infeasible individuals
}



char fn[100];
sprintf(fn,"%sdist%d",output_dir,generation);
if (NEAT::printdist)
pop->print_distribution(fn);
//Average and max their fitnesses for dumping to file and snapshot
for (curspecies=(pop->species).begin(); curspecies!=(pop->species).end(); ++curspecies) {

//This experiment control routine issues commands to collect ave
//and max fitness, as opposed to having the snapshot do it,
//because this allows flexibility in terms of what time
//to observe fitnesses at

(*curspecies)->compute_average_fitness();
(*curspecies)->compute_max_fitness();
}

//Take a snapshot of the population, so that it can be
//visualized later on
//if ((generation%1)==0)
//  pop->snapshot();

//Only print to file every print_every generations
// if  (win||
//      ((generation%(NEAT::print_every))==0))
if (win && !firstflag)
{
for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
if ((*curorg)->winner) {
winnernum=((*curorg)->gnome)->genome_id;
cout<<"WINNER IS #"<<((*curorg)->gnome)->genome_id<<endl;
char filename[100];
sprintf(filename,"%s_winner", output_dir);
(*curorg)->print_to_file(filename);
}
}
firstflag = true;
}

//writing out stuff
if (generation%NEAT::print_every == 0 )
{
char filename[100];
sprintf(filename,"%s_record.dat",output_dir);
char fname[100];
sprintf(fname,"%s_archive.dat",output_dir);
archive.Serialize(fname);
//Record.serialize(filename);
sprintf(fname,"%sgen_%d",output_dir,generation);
pop->print_to_file_by_species(fname);
}
if(!speciation)
if (NEAT::multiobjective) {
for (curorg=measure_pop.begin(); curorg!=measure_pop.end(); curorg++) delete (*curorg);
//clear the old population
measure_pop.clear();
if (generation!=maxgens)
for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
measure_pop.push_back(new Organism(*(*curorg),true));
}
}
//Create the next generation

pop->epoch(generation);


return win;
}
*/
