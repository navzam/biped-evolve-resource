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
#include "alps.h"
#define NF_COGSAMPSQ 1
#define NF_RANDOM 2
#define NF_TRAIT 4

class Controller;
class Creature;
void create_world(Controller* controller,bool log=false,bool b=false);
void destroy_world();
void initialize_biped(const char* nefile);

float walker_novelty_metric(noveltyitem* x,noveltyitem* y);

Population *biped_generational(char* outputdir,const char *genes, int gens,bool novelty);
//int biped_generational_epoch(Population **pop2,int generation,data_rec& Record, noveltyarchive& archive, bool novelty);
int biped_generational_epoch(population_state* p, int gen);

int biped_novelty_realtime_loop(Population *pop,bool novelty);
int biped_epoch(NEAT::Population *pop,bool novelty=false);

Population *biped_novelty_realtime(char* outputdir,int par,const char* genes,bool novelty);
void evolvability_biped(Organism* org,char* fn,int* d=NULL,double *e=NULL,bool recall=false);
	
void biped_neat_initeval(NEAT::Population *pop,bool novelty=false);
NEAT::Population *biped_neat_init(bool novelty=false);
population_state* create_biped_popstate(char* outputdir,const char *genes, int gens,bool novelty);
Population *biped_alps(char* output_dir, const char *genes, int gens, bool novelty);
int biped_success_processing(population_state* pstate);


noveltyitem* biped_evaluate(NEAT::Organism* org,data_record* data=NULL);
dReal evaluate_controller(Controller* controller,noveltyitem* ni=NULL,data_record* record=NULL,bool log=false);

void simulationStep(const bool bMoviePlay = false);

extern vector<dGeomID> geoms;
extern vector<Creature*> creatures;
extern int novelty_function;


class Controller
{
public:
  vector<dReal> outs;
  int size;
  bool scale;
  bool debug;
  ofstream* dbgfile;
  Controller() { }
  Controller(int s,bool deb=false)
  {
    debug=deb;
    scale=true;
    size=s;
    for (int x=0; x<size; x++)
      outs.push_back(0.0);
    if (debug)
      dbgfile=new ofstream("controller.log");
  }
  virtual void update(double time,vector<dReal> sensors) {
    if (debug)
    {
      for (int x=0; x<size; x++)
      {
        *dbgfile << outs[x] << " ";
      }
      *dbgfile << "\n";
    }
  }
  virtual vector<dReal>* get_outputs()
  {
    return &outs;
  }
  virtual ~Controller()
  {
    if (debug)
      delete dbgfile;
  }
};

class CTRNNController: public Controller
{
public:
  NEAT::Network* net;
  NEAT::Genome* genes;
  CTRNNController(NEAT::Network* n,NEAT::Genome *g=NULL,bool dbg=false):Controller(6,dbg)
  {
    scale=true;
    net=n;
    genes=g;
    n->init_ctrnn();
    for (int x=0; x<50; x++)
      net->activate_ctrnn(0.02);
    for (int x=0; x<size; x++)
      outs[x]=net->outputs[x]->output;
  }
  virtual void update(double time,vector<dReal> sensors)
  {
    double sens[20];
    for (int x=0; x<sensors.size(); x++)
      sens[x]=sensors[x];
    net->load_sensors(sens);
    net->activate_ctrnn(time);
    for (int x=0; x<6; x++)
    {
      outs[x]=net->outputs[x]->output;
    }
    Controller::update(time,sensors);
  }
};

class Creature
{
public:
  bool movie_rec;
  bool movie_play;

  ofstream* movie;
  ofstream* movie_rot;

  ifstream* movie_in;
  ifstream* movie_rot_in;

  Controller* controller;
  vector<dGeomID> geoms;
  vector<dBodyID> bodies;
  vector<bool> onground;
  vector<dJointID> joints;

  vector<dReal> current_angles;
  vector<dReal> desired_angles;

  vector<dReal> lo_limit;
  vector<dReal> hi_limit;

  vector<dReal> sensors;
  vector<dReal> desired_angvel;
  vector<dReal> delta_angles;

  vector<dReal> p_terms;
  vector<dReal> d_terms;

  dWorldID world;
  dSpaceID space;
  dVector3 pos;

  Creature(bool mov=false,bool play=false) {
    if (play)
      mov=false;
    movie_play=play;
    movie_rec=mov;
    if (play)
    {
      movie_in=new ifstream("movie_pos.dat");
      movie_rot_in=new ifstream("movie_rot.dat");
    }
    if (mov)
    {
      movie=new ofstream("movie_pos.dat");
      movie_rot=new ofstream("movie_rot.dat");
    }

  }

  virtual void Update(double timestep)
  {
    // If recording, record positions and rotations
    if(movie_rec)
    {
      for(int x = 0; x < geoms.size(); ++x)
      {
        dQuaternion rot;
        const dReal *pos = dGeomGetPosition(geoms[x]);
        dGeomGetQuaternion(geoms[x], rot);
        *movie << pos[0] << " " << pos[1] << " " << pos[2] << " ";
        *movie_rot << rot[0] << " " << rot[1] << " " << rot[2] << " " << rot[3] << " ";
      }
      *movie << endl;
      *movie_rot << endl;
    }
    
    // If playing, set positions and rotations
    if(movie_play)
    {
      for(int k = 0; k < geoms.size(); ++k)
      {
        dReal x, y, z;
        dQuaternion q;
        *movie_in >> x >> y >> z;
        *movie_rot_in >> q[0] >> q[1] >> q[2] >> q[3];
        dGeomSetPosition(geoms[k], x, y, z);
        dGeomSetQuaternion(geoms[k], q);
      }
    }
  }

  dReal TotalMass()
  {
    dReal total_mass=0.0;
    for (int x=0; x<bodies.size(); x++)
    {
      dMass m;
      dBodyGetMass(bodies[x],&m);
      total_mass+=m.mass;
    }
    return total_mass;
  }

  void CenterOfMass(dVector3 center)
  {
    dReal total_mass=0.0;
    dVector3 accum={0.0,0.0,0.0};
    const dReal* bpos;
    for (int x=0; x<bodies.size(); x++)
    {
      dMass m;
      dBodyGetMass(bodies[x],&m);
      bpos=dBodyGetPosition(bodies[x]);
      total_mass+=m.mass;
      for (int y=0; y<3; y++)
        accum[y]+=m.mass*bpos[y];
    }

    for (int x=0; x<3; x++)
      center[x]=accum[x]/total_mass;
  }

  virtual void Create(dWorldID worldi, dSpaceID spacei, dVector3 posi,Controller* cont)
  {
    controller=cont;
    world=worldi;
    space=spacei;
    pos[0]=posi[0];
    pos[1]=posi[1];
    pos[2]=posi[2];
  }

  virtual void Destroy()
  {
    for (int x=0; x<geoms.size(); x++)
      dGeomDestroy(geoms[x]);
  }

  virtual bool abort()=0;
  virtual dReal fitness()=0;
  int add_fixed(int b1,int b2)
  {
    dBodyID bd1=bodies[b1];
    dBodyID bd2;
    if (b2!=(-1))
    {
      bd2=bodies[b2];
    }
    else
    {
      bd2=0;
    }
    dJointID tempjoint = dJointCreateFixed(world,0);
    dJointAttach(tempjoint,bd1,bd2);
    dJointSetFixed(tempjoint);
    joints.push_back(tempjoint);
    return joints.size()-1;
  }
  int add_hinge(int b1,int b2,dVector3 anchor,dVector3 axis,dReal lostop, dReal histop, dReal fmax)
  {
    dJointID tempjoint = dJointCreateHinge(world,0);
    dJointAttach(tempjoint,bodies[b1],bodies[b2]);
    dJointSetHingeAnchor(tempjoint,pos[0]+anchor[0],pos[1]+anchor[1],pos[2]+anchor[2]);
    dJointSetHingeAxis(tempjoint,axis[0],axis[1],axis[2]);
    dJointSetHingeParam(tempjoint,dParamLoStop,lostop);
    dJointSetHingeParam(tempjoint,dParamHiStop,histop);
    dJointSetHingeParam(tempjoint,dParamFMax,fmax);
    joints.push_back(tempjoint);

    return joints.size()-1;
  }

  int add_universal(int b1,int b2,dVector3 anchor, dVector3 axis1, dVector3 axis2, dReal lostop1, dReal histop1, dReal lostop2, dReal histop2, dReal fmax1, dReal fmax2)
  {
    dJointID tempjoint = dJointCreateUniversal(world,0);
    dJointAttach(tempjoint,bodies[b1],bodies[b2]);
    dJointSetUniversalAnchor(tempjoint,pos[0]+anchor[0],pos[1]+anchor[1],pos[2]+anchor[2]);
    dJointSetUniversalAxis1(tempjoint,axis1[0],axis1[1],axis1[2]);
    dJointSetUniversalAxis2(tempjoint,axis2[0],axis2[1],axis2[2]);
    dJointSetUniversalParam(tempjoint,dParamLoStop,lostop1);
    dJointSetUniversalParam(tempjoint,dParamHiStop,histop1);
    dJointSetUniversalParam(tempjoint,dParamLoStop2,lostop2);
    dJointSetUniversalParam(tempjoint,dParamHiStop2,histop2);
    dJointSetUniversalParam(tempjoint,dParamFMax,fmax1);


    dJointSetUniversalParam(tempjoint,dParamFMax2,fmax2);

    joints.push_back(tempjoint);
    return joints.size()-1;
  }

  int add_sphere(dReal density, dReal radius, const dVector3 p)
  {

    dBodyID tempbody;
    dGeomID tempgeom;
    dMass m;
    tempbody = dBodyCreate (world);
    dMassSetSphereTotal(&m,density,radius);
    tempgeom = dCreateSphere(0,radius);
    dGeomSetBody(tempgeom,tempbody);
    dBodySetMass(tempbody,&m);
    dBodySetPosition(tempbody,pos[0]+p[0],pos[1]+p[1],pos[2]+p[2]);
    dSpaceAdd(space,tempgeom);

    bodies.push_back(tempbody);
    geoms.push_back(tempgeom);
    onground.push_back(false);
    return bodies.size()-1;

  }

  int add_cylinder(int axis, dReal density,dReal length, dReal radius, const dVector3 p,dBodyID* k=NULL)
  {
    dReal a[]={0.0,0.0,0.0};

    dBodyID tempbody;
    dGeomID tempgeom;
    dMass m;

    tempbody = dBodyCreate (world);
    if (k!=NULL)
      (*k)=tempbody;
    dQuaternion q;
    if (axis==1)
    {
      a[1]=1.0;
    }
    else if (axis==2)
    {
      a[0]=1.0;
    }
    else
    {
      a[2]=1.0;
    }
    dQFromAxisAndAngle (q,a[0],a[1],a[2], M_PI * 0.5);
    dBodySetQuaternion (tempbody,q);
    dMassSetCylinderTotal (&m,density,axis,radius,length);
    dBodySetMass (tempbody,&m);
    tempgeom = dCreateCylinder(0, radius, length);
    dGeomSetBody (tempgeom,tempbody);
    dBodySetPosition (tempbody, pos[0]+p[0],pos[1]+p[1], pos[2]+p[2]);
    dSpaceAdd (space, tempgeom);

    geoms.push_back(tempgeom);
    bodies.push_back(tempbody);
    onground.push_back(false);
    return bodies.size()-1;
  }
  virtual ~Creature()
  {

    if (movie_rec)
    {
      delete movie;
      delete movie_rot;
      //cout << "terminating.." << endl;
    }
    if (movie_play)
    {
      delete movie_in;
      delete movie_rot_in;
    }

  }
};

//BIPED PARAMETERS
static dReal SCALE_FACTOR = 3;
static dReal FOOTX_SZ =1.0/SCALE_FACTOR;
static dReal FOOTY_SZ =0.5/SCALE_FACTOR;
static dReal FOOTZ_SZ =1.0/SCALE_FACTOR;
static dReal LLEG_LEN =1.0/SCALE_FACTOR;
static dReal LLEG_RAD =0.2/SCALE_FACTOR;
static dReal ULEG_LEN =1.0/SCALE_FACTOR;
static dReal ULEG_RAD =0.2/SCALE_FACTOR;
static dReal TORSO_LEN =1.0/SCALE_FACTOR;
static dReal TORSO_RAD =0.3/SCALE_FACTOR;
static dReal ORIG_HEIGHT= (TORSO_RAD/2.0+ULEG_LEN+LLEG_LEN+FOOTZ_SZ);
static dReal DENSITY=0.5;
static dReal TORSO_DENSITY=1.0;
static dReal FOOT_DENSITY=0.1;
extern dReal MAXTORQUE_KNEE;//= 5.0;
extern dReal MAXTORQUE_HIPMINOR;//= 5.0;
extern dReal MAXTORQUE_HIPMAJOR;//= 5.0;
extern dReal P_CONSTANT; //= 9.0;
extern dReal D_CONSTANT; //= 0.0;

class Biped: public Creature
{
public:
  int step;
  dVector3 orig_com;
  dVector3 orig_left;
  dVector3 orig_right;
  dVector3 curr_com;
  bool log;
  ofstream* logfile;
  dJointFeedback feedback[6];

  //keeping track of foot positionxorz
  bool leftdown;
  bool rightdown;

  vector<float> lft; //left foot time
  vector<float> lfx; //x
  vector<float> lfy; //y
  vector<float> rft; //right foot time
  vector<float> rfx; //x
  vector<float> rfy; //y

  Biped(bool logging=false,bool movie=false):Creature(logging,movie) {
    step=0;

    leftdown=false;
    rightdown=false;

    log=logging;

    if (log)
    {
      logfile=new ofstream("log.dat");
    }

    for (int x=0; x<6; x++)
    {
      p_terms.push_back(P_CONSTANT);
      d_terms.push_back(D_CONSTANT);
      desired_angles.push_back(0.0);
      current_angles.push_back(0.0);
      delta_angles.push_back(0.0);
      desired_angvel.push_back(0.0);
      lo_limit.push_back(0.0);
      hi_limit.push_back(0.0);
    }

    for (int x=0; x<8; x++)
      sensors.push_back(0.0);
  }
    
  int add_foot(dReal density, dReal radius, const dVector3 p)
  {

    dBodyID tempbody;
    dGeomID maingeom[3];
    dGeomID tempgeom[3];

    dMass m;
    tempbody = dBodyCreate (world);
    dMassSetSphereTotal(&m,density,radius);
    dBodySetPosition(tempbody,pos[0]+p[0],pos[1]+p[1],pos[2]+p[2]);
    bodies.push_back(tempbody);
    onground.push_back(false);

    for (int x=0; x<3; x++)
    {
      maingeom[x] = dCreateGeomTransform(space);
      tempgeom[x] = dCreateSphere(0,radius);

      dGeomSetBody(maingeom[x],tempbody);

      dGeomTransformSetGeom(maingeom[x],tempgeom[x]);
      dGeomTransformSetCleanup(maingeom[x],1);
      dGeomTransformSetInfo(maingeom[x],1);

      geoms.push_back(maingeom[x]);

      if (x==0)
        dGeomSetPosition(tempgeom[x],0.07,-0.08,0.0);
      if (x==1)
        dGeomSetPosition(tempgeom[x],0.07,0.08,0.0);
      if (x==2)
        dGeomSetPosition(tempgeom[x],-0.20,0.0,0.0);

    }
    return bodies.size()-1;
  }


  virtual dReal fitness()
  {
    dVector3 new_com;
    CenterOfMass(new_com);
    double fitness=0.0;
    for (int x=0; x<2; x++)
    {
      double delta=new_com[x]-orig_com[x];
      delta*=delta;
      fitness+=delta;
    }
    //fitness=sqrt(fitness);
    return fitness;
  }

  ~Biped()
  {
    if (log)
      delete logfile;
  }

  virtual bool abort()
  {
    const dReal *torsoPos = dBodyGetPosition(bodies[6]);
    if(torsoPos[2] < 0.5 * (ORIG_HEIGHT))
      return true;
    
    return false;
  }

  void create_leg(const dVector3 offset)
  {
    dVector3 xAxis = {1.0, 0.0, 0.0};
    dVector3 yAxis = {0.0, -1.0, 0.0};
    dVector3 zAxis = {0.0, 0.0, 1.0};

    dVector3 p = {offset[0], offset[1], offset[2]};

    dVector3 foot_pos = {p[0], p[1], p[2] + (FOOTZ_SZ / 2.0)};

    int foot = add_sphere(FOOT_DENSITY, FOOTZ_SZ / 2.0, foot_pos);

    dVector3 lower_pos = {p[0], p[1], p[2] + FOOTZ_SZ + LLEG_LEN / 2.0};
    int lowerleg = add_cylinder(3, DENSITY, LLEG_LEN, LLEG_RAD, lower_pos);
    dVector3 upper_pos = {p[0], p[1], p[2] + FOOTZ_SZ + LLEG_LEN + ULEG_LEN / 2.0};
    int upperleg = add_cylinder(3, DENSITY, ULEG_LEN, ULEG_RAD, upper_pos);

    dVector3 foot_joint_a = {p[0], p[1], p[2] + FOOTZ_SZ};
    dVector3 knee_joint_a = {p[0], p[1], p[2] + FOOTZ_SZ + LLEG_LEN};

    add_fixed(foot, lowerleg);
    add_hinge(lowerleg, upperleg, knee_joint_a, yAxis, -1.4, 0.0, MAXTORQUE_KNEE);
  }

  virtual void Create(dWorldID worldi, dSpaceID spacei, dVector3 posi,Controller* cont)
  {
    dVector3 xAxis={1.0,0.0,0.0};
    dVector3 nxAxis={-1.0,0.0,0.0};
    dVector3 yAxis={0.0,1.0,0.0};
    dVector3 zAxis={0.0,0.0,1.0};

    Creature::Create(worldi,spacei,posi,cont);
    dVector3 leftLegPos={0.0,0.0,0.0};
    dVector3 rightLegPos={0.0,TORSO_LEN+ULEG_RAD,0.0};
    dVector3 torsoPos={0.0,(TORSO_LEN+ULEG_RAD)/2.0,ULEG_LEN+LLEG_LEN+FOOTZ_SZ};


    dVector3 leftHip={leftLegPos[0],leftLegPos[1]+ULEG_RAD,torsoPos[2]};
    dVector3 rightHip={rightLegPos[0],rightLegPos[1]-ULEG_RAD,torsoPos[2]};

    create_leg(leftLegPos);
    create_leg(rightLegPos);

    int torso=add_cylinder(2,TORSO_DENSITY,TORSO_LEN,TORSO_RAD,torsoPos);
    //-1.4,1.4
    add_universal(torso,2,leftHip,xAxis,yAxis,-0.8,0.8,-1.3,1.6,MAXTORQUE_HIPMINOR,MAXTORQUE_HIPMAJOR);
    add_universal(torso,5,rightHip,nxAxis,yAxis,-0.8,0.8,-1.3,1.6,MAXTORQUE_HIPMINOR,MAXTORQUE_HIPMAJOR);

    lo_limit[0]=dJointGetHingeParam(joints[1],dParamLoStop);
    lo_limit[1]=dJointGetHingeParam(joints[3],dParamLoStop);
    hi_limit[0]=dJointGetHingeParam(joints[1],dParamHiStop);
    hi_limit[1]=dJointGetHingeParam(joints[3],dParamHiStop);

    lo_limit[2]=dJointGetUniversalParam(joints[4],dParamLoStop);
    lo_limit[3]=dJointGetUniversalParam(joints[5],dParamLoStop);
    hi_limit[2]=dJointGetUniversalParam(joints[4],dParamHiStop);
    hi_limit[3]=dJointGetUniversalParam(joints[5],dParamHiStop);

    lo_limit[4]=dJointGetUniversalParam(joints[4],dParamLoStop2);
    lo_limit[5]=dJointGetUniversalParam(joints[5],dParamLoStop2);
    hi_limit[4]=dJointGetUniversalParam(joints[4],dParamHiStop2);
    hi_limit[5]=dJointGetUniversalParam(joints[5],dParamHiStop2);

    dJointSetFeedback(joints[1],&feedback[0]);
    dJointSetFeedback(joints[3],&feedback[1]);
    dJointSetFeedback(joints[4],&feedback[2]);
    dJointSetFeedback(joints[5],&feedback[3]);
    dJointSetFeedback(joints[0],&feedback[4]);
    dJointSetFeedback(joints[2],&feedback[5]);


    CenterOfMass(orig_com);
    CenterOfMass(curr_com);
    orig_left[0]=dBodyGetPosition(bodies[0])[0];
    orig_right[0]=dBodyGetPosition(bodies[0])[1];
    orig_left[1]=dBodyGetPosition(bodies[3])[0];
    orig_right[1]=dBodyGetPosition(bodies[3])[1];
    orig_left[2]=0.0;
    orig_right[2]=0.0;
  }

  void print_behavior()
  {

    cout << "LEFTFOOTSTEPS: " << lft.size() << endl;
    cout << "RIGHTFOOTSTEPS: " << rft.size() << endl;

    for (int x=0; x<lft.size(); x++)
    {
      cout << "LFT " << x << " time: " << lft[x] << " x: " << lfx[x] << " y: " << lfy[x] << endl;
    }
    for (int x=0; x<rft.size(); x++)
    {
      cout << "RFT " << x << " time: " << rft[x] << " x: " << rfx[x] << " y: " << rfy[x] << endl;
    }

  }
  void crossproduct(double* a,double *b,double *r) {
    r[0]=a[1]*b[2]-a[2]*b[1];
    r[1]=a[2]*b[0]-a[0]*b[2];
    r[2]=a[0]*b[1]-a[1]*b[0];
  }
   
  virtual void Update(double timestep)
  {
    int torso=6;
    const dReal* res=dBodyGetAngularVel(bodies[torso]);
    dVector3 p1;
    dVector3 p2;
    dVector3 cp;
    dBodyGetRelPointPos(bodies[torso],0,0,0,p1);
    dBodyGetRelPointPos(bodies[torso],0,1,0,p2);
    for(int k=0;k<3;k++)
      p2[k]=p2[k]-p1[k];
	
    //cout << p2[0] << " " << p2[1] << " " << p2[2] << endl;
    //dReal upVector[3]={.707,0,.707};	
    //dReal upVector[3]={0.587,0,0.866};	
    dReal upVector[3]={0.0,0,1.0};	
    crossproduct(p2,upVector,cp);
    double u_factor=2.0 * (1.0-NEAT::gravity);
    dReal utorque[3]={u_factor*cp[0],u_factor*cp[1],u_factor*cp[2]};
    //cout << res[0] << " " << res[1] << " " << res[2] << endl;
    double k_factor=-0.3 * (1.0-NEAT::gravity);
    dReal damptorque[3]={k_factor*res[0],k_factor*res[1],k_factor*res[2]};
    if(NEAT::gravity!=1.0) {
      dBodyAddTorque(bodies[torso],utorque[0],utorque[1],utorque[2]);
      dBodyAddTorque(bodies[torso],damptorque[0],damptorque[1],damptorque[2]);
      dBodyAddForce(bodies[torso],0,0,20.0*(1.0-NEAT::gravity));	
    }

    Creature::Update(timestep);
    if (movie_play)
      return;
    dReal old_angles[10];

    step++;

    for (int x=0; x<6; x++)
      old_angles[x]=current_angles[x];

    //read current angles
    current_angles[0]=dJointGetHingeAngle(joints[1]); //left knee
    current_angles[1]=dJointGetHingeAngle(joints[3]); //right knee

    current_angles[2]=dJointGetUniversalAngle1(joints[4]); //left outhip
    current_angles[3]=dJointGetUniversalAngle1(joints[5]); //right outhip

    current_angles[4]=dJointGetUniversalAngle2(joints[4]); //left mainhip
    current_angles[5]=dJointGetUniversalAngle2(joints[5]); //right mainhip


    for (int x=0; x<6; x++)
      delta_angles[x]=(current_angles[x]-old_angles[x])/timestep;

    //record behavior
    bool newleftdown=onground[0];
    bool newrightdown=onground[3];

    if (newleftdown)
      sensors[0]=1.0;
    else
      sensors[0]=0.0;

    if (newrightdown)
      sensors[1]=1.0;
    else
      sensors[1]=0.0;

    //update controller
    controller->update(timestep,sensors);
    vector<dReal>* outs=controller->get_outputs();

    //calculate values to pass to joint motors
    if (log)
      (*logfile) << "-PID" <<endl;



    for (int x=0; x<6; x++)
    {
      desired_angles[x]=(*outs)[x];

      if (controller->scale)
      {
        if (desired_angles[x]>1.0) desired_angles[x]=1.0;
        if (desired_angles[x]<0.0) desired_angles[x]=0.0;
        desired_angles[x]= lo_limit[x]+(hi_limit[x]-lo_limit[x])*desired_angles[x];
      }
    }

    for (int x=0; x<6; x++)
    {
      dReal delta=desired_angles[x]-current_angles[x];


      double p_term = p_terms[x]* delta;
      double d_term = (-d_terms[x]*delta_angles[x]);
      desired_angvel[x]=p_term+d_term;
      if (log)
        (*logfile) << p_term << " " << d_term << " " << desired_angvel[x]
          << " " << delta_angles[x] << " " << desired_angles[x] << " " << current_angles[x] << endl;
    }

    if (log)
      (*logfile) << "-FEED" <<endl;

    if (log)
      for (int x=0; x<4; x++)
    {
      for (int k=0; k<3; k++)
        (*logfile) << feedback[x].f1[k] << " ";
      for (int k=0; k<3; k++)
        (*logfile) << feedback[x].f2[k] << " ";
      (*logfile) << endl;
    }

    //update joint motors
    dJointSetHingeParam(joints[1],dParamVel,desired_angvel[0]); //left knee
    dJointSetHingeParam(joints[3],dParamVel,desired_angvel[1]); //right knee

    dJointSetUniversalParam(joints[4],dParamVel,desired_angvel[2]); //left hipout
    dJointSetUniversalParam(joints[5],dParamVel,desired_angvel[3]); //right hipout

    dJointSetUniversalParam(joints[4],dParamVel2,desired_angvel[4]); //left hipmain
    dJointSetUniversalParam(joints[5],dParamVel2,desired_angvel[5]); //right hipmain


    if (!leftdown && newleftdown)
    {
      if (lft.size()==0 || (step-lft[lft.size()-1] > 100 ))
      {
        CenterOfMass(curr_com);
        lft.push_back(step);
        lfx.push_back(curr_com[0]);
        lfy.push_back(curr_com[1]);
      }
    }
    if (!rightdown && newrightdown)
    {
      if (rft.size()==0 || (step-rft[rft.size()-1] > 100 ))
      {

        CenterOfMass(curr_com);
        rft.push_back(step);
        rfx.push_back(curr_com[0]);
        rfy.push_back(curr_com[1]);
      }
    }

    leftdown=newleftdown;
    rightdown=newrightdown;
    //reset ground sensors for feetz
    onground[0]=false;
    onground[3]=false;

  }
};