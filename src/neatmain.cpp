//#include <mcheck.h>
//#include <google/heap-profiler.h>
#include <iostream>
#include <fstream>
using namespace std;
#include <tclap/CmdLine.h>
#include <cstring>

#include <vector>
#include <unistd.h>	
#include "neat.h"
#include "genome.h"
#include "population.h"
#include "experiments.h"
#include "biped.h"
#include "alps.h"
using namespace TCLAP;
CmdLine cmd("Maze evolution", ' ', "0.1");

int main(int argc, char **argv) {
  //HeapProfilerStart("profout");
  // mtrace(); 
  ValueArg<float> grav("","gravity","Gravity setting",false,1.0,"float");
  cmd.add(grav);

  ValueArg<string> genes("z","sg","Starter genes",false,"mazestart_orig","string");
  cmd.add(genes);

  ValueArg<string> settings("s","settings","Settings file",false,"maze.ne","string");
  cmd.add(settings);
 
  ValueArg<string> output("o","output","Output directory",false,"./results","string");
  cmd.add(output);

  ValueArg<string> seed_genome("c","seed","Seed Genome",false,"","string");
  cmd.add(seed_genome);

  SwitchArg age_objective("","ageobj","age objective",false);
  cmd.add(age_objective);

  SwitchArg passive_switch("","passive","passive search",false);
  cmd.add(passive_switch);

  SwitchArg local_switch("","lc","Local competition",false);
  cmd.add(local_switch);
  
  SwitchArg multiobj_switch("","mo","Multiobjective",false);
  cmd.add(multiobj_switch);
 
  SwitchArg remove_random("","remrand","Remove random individuak",false);
  cmd.add(remove_random); 

  SwitchArg extinction("","extinct","Turn on random extinctions",false);
  cmd.add(extinction);

  SwitchArg alpsmode("","alps","ALPs-type dealie",false);
  cmd.add(alpsmode);
  
  SwitchArg mo_speciation("","mos","Multiobjective speciation",true);
  cmd.add(mo_speciation);

  SwitchArg noveltySwitch("n","novelty","Enable novelty search",false);
  cmd.add(noveltySwitch);

  SwitchArg evaluateSwitch("","eval","Evaluate a genome",false);
  cmd.add(evaluateSwitch);

  SwitchArg constraintSwitch("","constraint","Enable constraint-based NS",false);
  cmd.add(constraintSwitch);

  SwitchArg generationalSwitch("","gen","Enable generational search",false);
  cmd.add(generationalSwitch);

  SwitchArg mcSwitch("","mc","Enable minimal criteria",false);
  cmd.add(mcSwitch);

  ValueArg<string> nov_measure("","nm","Novelty Measure",false,"std","string");
  cmd.add(nov_measure);

  ValueArg<string> fit_measure("f","fm","Fitness Measure",false,"goal","string");
  cmd.add(fit_measure);

  ValueArg<int> extra_param("p","parameter","Extra Parameter",false,0,"int");
  cmd.add(extra_param);

  ValueArg<int> num_samples("","samples","Num Samples",false,1,"int");
  cmd.add(num_samples);

  ValueArg<int> generation_arg("","gens","Num generations",false,1000,"int");
  cmd.add(generation_arg);

  ValueArg<int> rng_seed("r","random_seed","Random Seed",false,-1,"int");
  cmd.add(rng_seed);

  ValueArg<long long> netindex("","ni","Net Index",false,0,"long int");
  cmd.add(netindex);

  cmd.parse(argc,argv);
  char filename[100]="./runoutput_";
  char settingsname[100]="maze.ne";
  char startgenes[100]="mazestartgenes";
  int param;
  int generations=generation_arg.getValue();
  NEAT::Population *p;

  //***********RANDOM SETUP***************//
  /* Seed the random-number generator with current time so that
  the numbers will be different every time we run.    */
  srand( (unsigned)time( NULL )  + getpid());
 
  if(rng_seed.getValue()!=-1)
    srand((unsigned)rng_seed.getValue());

  strcpy(settingsname,settings.getValue().c_str());
  strcpy(filename,output.getValue().c_str());
  strcpy(startgenes,genes.getValue().c_str());

  NEAT::load_neat_params(settingsname,true);
  
  NEAT::gravity=grav.getValue();

  NEAT::mo_speciation=mo_speciation.getValue();
  if(!NEAT::mo_speciation) {
    cout << "speciation off" << endl;
    NEAT::speciation=mo_speciation.getValue();
  }

  if(local_switch.getValue()) 
    NEAT::local_competition=true;
 
  if(multiobj_switch.getValue())
    NEAT::multiobjective=true;
  param = extra_param.getValue();
  cout<<"loaded"<<endl;

  cout << "Start genes: " << startgenes << endl;
  cout << "Generations: " << generations << endl; 
  set_age_objective(age_objective.getValue()); 
  if(age_objective.getValue() || alpsmode.getValue()) {
    NEAT::fresh_genetic_prob=0.05;
  }
  set_evaluate(evaluateSwitch.getValue());
  set_extinction(extinction.getValue());
  set_fit_measure(fit_measure.getValue());
  set_nov_measure(nov_measure.getValue());
  set_random_replace(remove_random.getValue());

  cout << "Num Samples: " << num_samples.getValue() << endl;
  set_samples(num_samples.getValue());

  set_seed(seed_genome.getValue()); 

  cout << "Minimal criteria engaged? " << mcSwitch.getValue() << endl;
  set_minimal_criteria(mcSwitch.getValue());

  /*
  long long netindex_val=netindex.getValue(); 
  enumerate_behaviors(mazename,netindex_val,filename,param);
  return 0;
  */
  
  set_constraint_switch(constraintSwitch.getValue());

  //p = classifier_generational(filename,mazename,param,startgenes,generations,noveltySwitch.getValue());
  //exit(0);

  if(!generationalSwitch.getValue())
  {
    p = biped_novelty_realtime(filename,param,startgenes,noveltySwitch.getValue());
  }
  else if(alpsmode.getValue())
  {
    p = biped_alps(filename,startgenes,50,noveltySwitch.getValue());
  }
  else
  {
    p = biped_generational(filename, startgenes, generations, noveltySwitch.getValue());
  }
  //HeapProfilerStop();

  return(0);
 
}
