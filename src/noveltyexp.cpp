#include "read_data.h"
#include "experiments.h"
#include "noveltyset.h"

#include "datarec.h"
#include "graph.h"

#include "histogram.h"
#include "calc_evol.h"
#include "genome.h"
//#define DEBUG_OUTPUT 1

#include <algorithm>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <math.h>

#include "population.h"
#include "population_state.h"
#include "alps.h"
//#include "modularity/modularity.hpp"

static vector<vector<float> > classifier_train_data;
static vector<vector<float> > classifier_valid_data;
static vector<vector<float> > classifier_test_data;
static ofstream *logfile=NULL;

static vector<float> best_fits;
plot front_plot;
plot fitness_plot;
plot behavior_plot;
//void modularity(Organism* org,char* fn);

/*double modularity_score(Genome* start_genome) {
Network* test_net=start_genome->genesis(0);
Graph* g=test_net->to_graph();    
std::vector<std::set<Graph::vertex_descriptor> > mods;
float modularity = mod::h_modules(*g,mods);
//cout << "modularity: " << modularity << endl;
return modularity;
delete test_net;
delete g;
}*/
using namespace std;
enum novelty_measure_type { novelty_fitness, novelty_sample, novelty_accum, novelty_sample_free };
static novelty_measure_type novelty_measure = novelty_sample;

enum fitness_measure_type { fitness_uniform,fitness_goal, fitness_drift, fitness_std,fitness_rnd,fitness_spin,fitness_changegoal,fitness_collisions,fitness_reachone ,fitness_aoi,fitness_collgoal};
static fitness_measure_type fitness_measure = fitness_goal;
bool age_objective=false;
bool population_dirty=false;

static bool regression=true;

static bool extinction=true;
bool get_age_objective() { return age_objective; }
void set_age_objective(bool ao) { age_objective=ao; }
void set_extinction(bool _ext) {
  extinction=_ext;
}
static int change_extinction_length=5;

static int change_goal_length=5;

static int number_of_samples = 1;
bool seed_mode = false;
char seed_name[100]="";
bool minimal_criteria=false;
bool evaluate_switch=false;

static bool constraint_switch=false;
static bool rand_repl=false;

void set_evaluate(bool val) {
  evaluate_switch=val;
}
void set_random_replace(bool val)
{
  rand_repl = val;
}

void  set_constraint_switch(bool val)
{
  constraint_switch=val;
}
void set_minimal_criteria(bool mc)
{
  minimal_criteria=mc;
}

void set_samples(int s)
{
  number_of_samples=s;
}

void set_seed(string s)
{
  strcpy(seed_name,s.c_str());
  if (strlen(seed_name)>0)
    seed_mode=true;
}

void set_fit_measure(string m)
{
  if (m=="uniform")
    fitness_measure=fitness_uniform;
  if (m=="reachone")
    fitness_measure=fitness_reachone;
  if (m=="rnd")
    fitness_measure=fitness_rnd;
  if (m=="std")
    fitness_measure=fitness_std;
  if (m=="drift")
    fitness_measure=fitness_drift;
  if (m=="goal")
    fitness_measure=fitness_goal;
  if (m=="spin")
    fitness_measure=fitness_spin;
  if (m=="changegoal")
    fitness_measure=fitness_changegoal;
  if (m=="collisions")
    fitness_measure=fitness_collisions;
  if (m=="aoi")
    fitness_measure=fitness_aoi;
  if (m=="nocollide_goal")
    fitness_measure=fitness_collgoal;
  cout << "Fitness measure " << fitness_measure << endl;
}

void set_nov_measure(string m)
{
  if (m=="fitness")
    novelty_measure=novelty_fitness;
  if (m=="std" || m=="sample")
    novelty_measure=novelty_sample;
  if (m=="accum")
    novelty_measure=novelty_accum;
  if (m=="sample_free")
    novelty_measure=novelty_sample_free;
  cout << "Novelty measure " << novelty_measure << endl;
}

char output_dir[200]="";


static int param=-1;
static int push_back_size = 20;

//novelty metric for maze simulation
float maze_novelty_metric(noveltyitem* x,noveltyitem* y)
{
  float diff = 0.0;
  for (int k=0; k<(int)x->data.size(); k++)
  {
    diff+=hist_diff(x->data[k],y->data[k]);
  }
  return diff;
}

  void mutate_genome(Genome* new_genome,bool traits)
  {
    Network* net_analogue;
    double mut_power=NEAT::weight_mut_power;
    static double inno=1000;
    static int id=1000;
    new_genome->mutate_link_weights(mut_power,1.0,GAUSSIAN);
    if(traits) {
      vector<Innovation*> innos;
      if (randfloat()<NEAT::mutate_node_trait_prob) {
        //cout<<"mutate_node_trait"<<endl;
        new_genome->mutate_node_parameters(NEAT::time_const_mut_power,NEAT::time_const_mut_prob,
        NEAT::bias_mut_power,NEAT::bias_mut_prob);
      }

      if (randfloat()<NEAT::mutate_add_node_prob) 
        new_genome->mutate_add_node(innos,id,inno);
      else if (randfloat()<NEAT::mutate_add_link_prob) {
        //cout<<"mutate add link"<<endl;
        net_analogue=new_genome->genesis(0);
        new_genome->mutate_add_link(innos,inno,NEAT::newlink_tries);
        delete net_analogue;
      }


      if(randfloat()<0.5)
        new_genome->mutate_random_trait();
      if(randfloat()<0.2)
        new_genome->mutate_link_trait(1);
    }

    return;
  }

  /*void modularity(Organism* org,char* fn) {
  bool solution=false;
  fstream file;
  file.open(fn,ios::app|ios::out);
  cout <<"Modularity..." << endl;
  // file << "---" <<  " " << org->winner << endl;
  float minx,maxx,miny,maxy;
  envList[0]->get_range(minx,miny,maxx,maxy);
  double ox,oy,fit;
  int nodes;
  int connections;
  float mod=modularity_score(org->gnome);
  Genome *new_gene= new Genome(*org->gnome);
  Organism *new_org= new Organism(0.0,new_gene,0);
  noveltyitem* nov_item = maze_novelty_map(new_org);
  fit=nov_item->fitness;
  nodes=new_org->net->nodecount();
  connections=new_org->net->linkcount();
  ox=nov_item->data[0][0];
  oy=nov_item->data[0][1];
  if (nov_item->fitness>340) solution=true;

  //HOW IT WAS:
  //points[i*2]=(nov_item->data[0][0]-minx)/(maxx-minx);
  //points[i*2+1]=(nov_item->data[0][1]-miny)/(maxy-miny);
  delete new_org;
  delete nov_item;
  //file << endl;
  file << mod << " " << ox << " " << oy << " " << nodes << " " <<connections << " " << fit << " " << solution << endl;
  file.close();
  return;
  }*/

  float classify(vector<float>& results,vector<vector<float> >& data,Network* net,bool debug=false,bool real_val=false) {
    float correct=0;

    results.clear();
    for(int i=0;i<data.size();i++) {
      vector<float> line=data[i];
      float c_output = line[line.size()-1];
      double inputs[50];

     
      for(int j=0;j<line.size()-1;j++) {
        inputs[j]=line[j];
        if(debug)
          cout << inputs[j] << " ";
      }     


      net->flush();
      net->load_sensors(inputs);

      for (int z=0; z<10; z++)
        net->activate();

      float routput=net->outputs[0]->activation;
      float output=routput;
     
      if(debug)
        cout << output << " " << c_output << endl;
      
      if(!regression) {
        if (routput>0.5) output=1.0;
        else output=0.0;
      }
     

      //if (output==c_output)
      //	correct+=1;

      float error=c_output-output;
      error*=error;
      correct+=error;

      if (real_val) output=routput;
      results.push_back(output);
    }
    //if(debug)
    //  net->print_links_tofile("outy.dat");

    return 1.0 - (correct/data.size());

  }

  void accum_vect(vector<float>& acc,vector<float>& add) {
    vector<float>::iterator it1=acc.begin();
    vector<float>::iterator it2=add.begin();
    while(it1!=acc.end()) {
      //cout << (*it2) <<endl;
      (*it1)+=(*it2);
      //cout << (*it2) <<endl;
      it1++;
      it2++;
    }

  }

  void scale_vect(vector<float>& v,float factor) {
    vector<float>::iterator it=v.begin();
    while(it!=v.end()) {
      (*it)*=factor;
      it++;
    }
  }

  void precalc_outputs(vector<vector<float> > &outputs, vector<vector<float> > data, vector<Organism*> orgs) {

    int orgs_size = orgs.size();

    for(int i=0;i<orgs_size;i++) {
      vector<float> results;
      float perf=classify(results,data,orgs[i]->gnome->genesis(0));
      outputs.push_back(results);
    }

  }

  //todo:optimize
  float classify_ensemble_precalc(vector<float>& results,vector<vector<float> >& outputs, vector<vector <float> > data, vector<int> p,bool print_out=false) {

    int c_ind = data[0].size()-1;
    float err=0.0;
    float var=0.0;

    for(int i=0;i<data.size();i++) {
      float ans = data[i][c_ind];
      float accum = 0.0f; 

      vector<float> one_row;
      for(int j=0;j<p.size();j++) {
        float out = outputs[p[j]][i];
        one_row.push_back(out);
        accum+=out;
      }

      float prediction=accum/p.size();
      for(int j=0;j<p.size();j++) {
        float delta=one_row[j]-prediction;
        var+=delta*delta;
      }


      results.push_back(prediction);
      float delta=ans-prediction;
      err+= (1.0-delta*delta);
    }
    if(print_out)
      cout << "variance:" << var/data.size() << endl;
    return err/data.size();
  }

  float classify_ensemble(vector<float>& results,vector<vector<float> >& data, vector<Organism*> p,bool print_out=false) {

    vector<vector<float> > outputs;
    precalc_outputs(outputs,data,p);
    vector<int> seq;

    for(int i=0;i<p.size();i++)
      seq.push_back(i);

    return classify_ensemble_precalc(results,outputs,data,seq,print_out);


    vector<float> r_temp;
    vector<float> r_accum;

    for(int i=0;i<data.size();i++)
      r_accum.push_back(0.0);

    float weight_total=0.0;
 
    for(int i=0;i<p.size();i++) {
      float weight = 1.0; //p[i]->noveltypoint->fitness;
      //  if(weight<0.9 && !((weight_total==0.0 && i==(p.size()-1))))
      //  continue;
      //weight=1.0;
      //  cout << weight << endl;
      Network *newnet = p[i]->gnome->genesis(0);

      r_temp.clear();
      classify(r_temp,data,newnet,false,false);
      scale_vect(r_temp,weight);
      accum_vect(r_accum,r_temp);  

      weight_total+=weight;
      delete(newnet);
    }

    scale_vect(r_accum,1.0/weight_total);

    float c_index = data[0].size()-1;
    float correct=0;
 
    results.clear();
    float disagree=0.0f;
    for(int i=0;i<r_accum.size();i++) {

      float c_output=data[i][c_index];
      float output=r_accum[i];

      float t_disagree=1.0-output;
      if(output<t_disagree)
        t_disagree=output;
      disagree+=t_disagree;

      if(!regression) {
        if(output>0.5) output=1.0;
        else output=0.0;
      }
  
      //cout << output << endl;

      float error=c_output-output;
      error*=error;
      correct+=error;
      //if(output==c_output)
      // correct+=1;
      results.push_back(output);

    }
    if(print_out)
      cout << "disagreement: " << disagree/data.size() << endl;
    return 1.0- ((float)correct)/data.size();
  }


  void choose_ensemble(vector<vector<float> >& data,vector<Organism*> orgs,vector<Organism*>& ens) {
    int orgs_size=orgs.size();

    vector<int> ens_ind;
    vector< vector< float> > outputs;

    precalc_outputs(outputs,data,orgs);

    for(int i=0;i<10;i++) {
      int best_index=0;
      float best_perf=0;
      for(int j=0;j<orgs_size;j++) {
        vector<float> res;
        ens_ind.push_back(j);
        float perf=classify_ensemble_precalc(res,outputs,data,ens_ind);
        if(perf>best_perf) {
          best_perf=perf;
          best_index=j;
        }
        ens_ind.pop_back(); 
      }
      cout << i << " " << best_index << endl;
      ens_ind.push_back(best_index);
      ens.push_back(orgs[best_index]);
    }

  }


  noveltyitem* classifier_novelty_map(Organism *org,data_record* record) {
    static int best = 0;
    noveltyitem *new_item = new noveltyitem;
    Network* net = org->net;
    new_item->genotype=new Genome(*org->gnome);
    new_item->phenotype=new Network(*org->net);
    vector< vector<float> > gather;

    double fitness=0.0001;
    static float highest_fitness=0.0;

    new_item->viable=true;


    gather.clear();
 
    //todo: optimize
    vector<float> classifications;
    vector<float> classifications_gen;
    fitness= classify(classifications,classifier_train_data,net) + 0.0001;
    float generalization = classify(classifications_gen,classifier_valid_data,net);

    gather.push_back(classifications_gen);
  
    /*
    //measure novelty by which inputs the ANN is listening to
    vector<float> listening;
    vector<int> l_nodes;
    
    for(int i=0;i<40;i++) 
    listening.push_back(0.0);
    
    net->listening_nodes(l_nodes); 
    
    float val=1.0/l_nodes.size();
    for(int i=0;i<l_nodes.size();i++) {
    int node=l_nodes[i]-1;
    listening[node]=val;
    }
    gather.push_back(listening);
    */

    if (fitness>highest_fitness)
      highest_fitness=fitness;

    for (int i=0; i<gather.size(); i++)
      new_item->data.push_back(gather[i]);

    new_item->fitness=fitness;
    new_item->secondary=fitness;
    org->fitness=fitness;

    return new_item;
  }

  static int maxgens;

  population_state* create_classifier_popstate(char* outputdir,const char* classfile,int param,const char *genes, int gens,bool novelty) {

    maxgens=gens;
    float archive_thresh=3.0;
    noveltyarchive *archive= new noveltyarchive(archive_thresh,*maze_novelty_metric,true,push_back_size,minimal_criteria,true);

    //if doing multiobjective, turn off speciation, TODO:maybe turn off elitism
    if (NEAT::multiobjective) NEAT::speciation=false;

    Population *pop;

    Genome *start_genome;
    char curword[20];
    int id;

    ostringstream *fnamebuf;
    int gen;

    if (!seed_mode)
      strcpy(seed_name,genes);
    if(seed_mode)
      cout << "READING IN SEED" << endl;
    ifstream iFile(seed_name,ios::in);
 
    char trainfile[200];
    char testfile[200];
    char validfile[200];
    sprintf(trainfile,"%s_train.dat",classfile);
    sprintf(testfile,"%s_test.dat",classfile);
    sprintf(validfile,"%s_valid.dat",classfile);
     
    classifier_train_data = read_classifier_data(trainfile);
    classifier_test_data = read_classifier_data(testfile);
    classifier_valid_data = read_classifier_data(validfile);

    if (outputdir!=NULL) strcpy(output_dir,outputdir);
    cout<<"START GENERATIONAL MAZE EVOLUTION"<<endl;

    cout<<"Reading in the start genome"<<endl;
    //Read in the start Genome
    iFile>>curword;
    iFile>>id;
    cout<<"Reading in Genome id "<<id<<endl;
    start_genome=new Genome(id,iFile);
    iFile.close();

    cout<<"Start Genome: "<<start_genome<<endl;

    //Spawn the Population
    cout<<"Spawning Population off Genome"<<endl;
    cout << "Start genomes node: " << start_genome->nodes.size() << endl;
    if(!seed_mode) 
      pop=new Population(start_genome,NEAT::pop_size);
    else
      pop=new Population(start_genome,NEAT::pop_size,0.0);
    cout<<"Verifying Spawned Pop"<<endl;
    pop->verify();
   
    //set evaluator
    pop->set_evaluator(&classifier_novelty_map);
    pop->evaluate_all();
    delete start_genome;
    return new population_state(pop,novelty,archive);

  }

  Population *classifier_generational(char* output_dir,const char* classfile,int param, const char *genes, int gens, bool novelty) {
    char logname[100];
    char fname[100];
    sprintf(logname,"%s_log.txt",output_dir);
    logfile=new ofstream(logname);
    //pop->set_compatibility(&behavioral_compatibility);    
    population_state* p_state = create_classifier_popstate(output_dir,classfile,param,genes,gens,novelty);
    
    for (int gen=0; gen<=maxgens; gen++)  { //WAS 1000
      cout<<"Generation "<<gen<<endl;
      bool win = classifier_generational_epoch(p_state,gen);
      p_state->pop->epoch(gen);

      if (win)
      {
        break;
      }

    }

    sprintf(fname,"%s_final",output_dir);
    p_state->pop->print_to_file_by_species(fname);
    delete logfile;
    delete p_state;
    return NULL;
  }

  void destroy_organism(Organism* curorg) {
    curorg->fitness = SNUM/1000.0;
    curorg->noveltypoint->reset_behavior();
    curorg->destroy=true;
  }

  int classifier_success_processing(population_state* pstate) {
    static int gen=0; 
    double& best_fitness = pstate->best_fitness;
    double& best_secondary = pstate->best_secondary;

    vector<Organism*>::iterator curorg;
    Population* pop = pstate->pop;
    //Evaluate each organism on a test
    int indiv_counter=0;
 
    Organism* cur_champ=NULL;
    float high_fit=0.0;

    for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {

      if ((*curorg)->noveltypoint->fitness > high_fit) {
        cur_champ=*curorg;
        high_fit=((*curorg)->noveltypoint->fitness);
      }

      if ((*curorg)->noveltypoint->fitness > best_fitness)
      {
        best_fitness = (*curorg)->noveltypoint->fitness;
        cout << "NEW BEST " << best_fitness << endl;
      }

      if (!pstate->novelty)
        (*curorg)->fitness = (*curorg)->noveltypoint->fitness;
    }

    if(logfile!=NULL)
      (*logfile) << pstate->generation*NEAT::pop_size<< " " << best_fitness << " " << best_secondary << " " << time(0) << endl;
   
    //vector<Organism*> orgs;
    vector<Organism*>& orgs=pop->organisms;
    vector<Organism*> ensemble;
    gen++;

    vector<float> ens_results;
    //choose_ensemble(classifier_valid_data,orgs,ensemble);

    cout << "CHAMP TRAIN PERF: " << classify(ens_results,classifier_train_data,cur_champ->gnome->genesis(0)) << endl;
    cout << "CHAMP TEST PERF: " << classify(ens_results,classifier_test_data,cur_champ->gnome->genesis(0)) << endl;
    //    cout << "ENSEMBLE TRAIN PERF: "  << classify_ensemble(ens_results,classifier_train_data,ensemble,true) << endl;

    //    cout << "ENSEMBLE TEST PERF: "  << 
    //    classify_ensemble(ens_results,classifier_test_data,ensemble,true) << endl;

    return 0;
  }

  int classifier_generational_epoch(population_state* pstate,int generation) {
    return 
      generalized_generational_epoch(pstate,generation,&classifier_success_processing);
  }

  int generalized_generational_epoch(population_state* pstate,int generation,successfunc success_processing) {
    pstate->generation++;

    bool novelty = pstate->novelty;
    noveltyarchive& archive = *pstate->archive;
    data_rec& Record = pstate->Record;
    Population **pop2 = &pstate->pop;
    Population* pop= *pop2;
    vector<Organism*>::iterator curorg,deadorg;
    vector<Species*>::iterator curspecies;
    vector<Organism*>& measure_pop=pstate->measure_pop;
    
    cout << "Number of genomes: " << Genome::increment_count(0) << endl;
    cout << "Number of genes: " << Gene::increment_count(0) << endl;
    cout << "Archive size: " << archive.get_set_size() << endl;

    if (NEAT::multiobjective) {  //merge and filter population	
      if(!novelty) NEAT::fitness_multiobjective=true;
      //if using alps-style aging
      if(pstate->max_age!=-1) 
      for (curorg=(measure_pop.begin());curorg!=measure_pop.end();++curorg) {
        (*curorg)->age++;	
        //if too old, delete
        if((*curorg)->age > pstate->max_age) {
          deadorg=curorg;
          if(pstate->promote!=NULL) {
            pstate->promote->measure_pop.push_back((*curorg));
          }
          else
            delete (*curorg);
          curorg=measure_pop.erase(deadorg);
          curorg--;
        }
      }

      for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
        measure_pop.push_back(new Organism(*(*curorg),true)); //TODO:maybe make a copy?
      }
        
      Genome* sg=pop->start_genome;
      evaluatorfunc ev=pop->evaluator; 
      delete pop;
      pop=new Population(measure_pop);
      pop->start_genome=sg;
      pop->set_evaluator(ev);
      *pop2= pop;
    }

    if (NEAT::evolvabilitytest)
      std::cout << "evol test still used...!!!!!!" << std::endl;

    int ret = success_processing(pstate);
    if(ret != 0)
      return 1;

    if (novelty)
    {

      archive.evaluate_population(pop,true);
      archive.evaluate_population(pop,false);

      pop->print_divtotal();


      for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
        if ( !(*curorg)->noveltypoint->viable && minimal_criteria)
        {
          (*curorg)->fitness = SNUM/1000.0;
          (*curorg)->noveltypoint->fitness = SNUM/1000.0;
          (*curorg)->noveltypoint->novelty = SNUM/1000.0;
        }
      }
      //cout << "ARCHIVE SIZE:" << archive.get_set_size() << endl;
      //cout << "THRESHOLD:" << archive.get_threshold() << endl;
      archive.end_of_gen_steady(pop);
    }
   
    if (NEAT::multiobjective) {
      archive.rank(pop->organisms);

      if (pop->organisms.size()>NEAT::pop_size) {
        //chop population down by half (maybe delete orgs that aren't used)
        int start=NEAT::pop_size; //measure_pop.size()/2;
        vector<Organism*>::iterator it;
        for (it=pop->organisms.begin()+start; it!=pop->organisms.end(); it++) {
          (*it)->species->remove_org(*it);
          delete (*it);
        }
        pop->organisms.erase(pop->organisms.begin()+start,pop->organisms.end());
      }
    }
    

	
#ifdef PLOT_ON11
    if(true) {
      vector<float> x,y,z;
      pop->gather_objectives(&x,&y,&z);
      front_plot.plot_data(x,y,"p","Pareto front");
      //best_fits.push_back(pstate->best_fitness);
      //fitness_plot.plot_data(best_fits,"lines","Fitness");

      /*
      vector<float> xc;
      vector<float> yc;
      for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {
      int sz=(*curorg)->noveltypoint->data[0].size();
      //xc.push_back((*curorg)->noveltypoint->data[0][sz-2]);
      //yc.push_back((*curorg)->noveltypoint->data[0][sz-1]);
      if((*curorg)->noveltypoint->viable) {
      xc.push_back((*curorg)->noveltypoint->data[0][sz-3]);
      yc.push_back((*curorg)->noveltypoint->data[0][sz-2]);
      }
      }
      behavior_plot.plot_data(xc,yc);

      }
      */
      vector<float> xc;
      vector<float> yc;
      float coltot=0.0;
      for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {

        coltot+=(*curorg)->datarec->ToRec[5];
        int sz=(*curorg)->noveltypoint->data[0].size();
        if((*curorg)->datarec->ToRec[5]>-5) {
          xc.push_back((*curorg)->noveltypoint->data[0][sz-2]);
          xc.push_back((*curorg)->noveltypoint->data[0][sz-1]);
        }
        else {
          yc.push_back((*curorg)->noveltypoint->data[0][sz-2]);
          yc.push_back((*curorg)->noveltypoint->data[0][sz-1]);
        }
      }
      cout << "COLTOT: " << coltot << endl;
      vector<vector <float> > blah;
      blah.push_back(xc);
      blah.push_back(yc);
      behavior_plot.plot_data_2d(blah);

    }
#endif

    // Print distribution (if enabled)
    if (NEAT::printdist)
    {
      char fn[100];
      sprintf(fn,"%sdist%d",output_dir,generation);
      pop->print_distribution(fn);
    }
    
    // For every species, ompute average and max fitnesses
    for (curspecies=(pop->species).begin(); curspecies!=(pop->species).end(); ++curspecies) {
      (*curspecies)->compute_average_fitness();
      (*curspecies)->compute_max_fitness();
    }

    // Write out stuff after a chunk of generations
    if ((generation+1)%NEAT::print_every == 0 )
    {
      char filename[100];
      sprintf(filename,"%s_record.dat",output_dir);
      char fname[100];
      sprintf(fname,"%s_archive.dat",output_dir);
      archive.Serialize(fname);
      sprintf(fname,"%sgen_%d",output_dir,generation);
      pop->print_to_file_by_species(fname);
        
      sprintf(fname,"%sfittest_%d",output_dir,generation);
      archive.serialize_fittest(fname);
    }

    if (NEAT::multiobjective) {
      for (curorg=measure_pop.begin(); curorg!=measure_pop.end(); curorg++) delete (*curorg);
      //clear the old population
      measure_pop.clear();
      if (generation!=maxgens)
      for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
        measure_pop.push_back(new Organism(*(*curorg),true));
      }
    }

    // Print average size and age
    pop->print_avg_age();

    return 0;
  }

