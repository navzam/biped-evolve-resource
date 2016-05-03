#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <list>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include "noveltyset.h"
#include "neat.h"
#include "network.h"
#include "population.h"
#include "organism.h"
#include "genome.h"
#include "species.h"
#include "datarec.h"
#include "population_state.h"

using namespace std;
using namespace NEAT;

void enumerate_behaviors(const char* name,long long parm,const char* outname,int count);
void mutate_genome(Genome* new_genome,bool traits=false);

void set_age_objective(bool ao); 
void set_evaluate(bool val);
void set_extinction(bool _ext);
void set_random_replace(bool val);
void set_constraint_switch(bool val);
void set_nov_measure(string m);
void set_fit_measure(string m);

void set_minimal_criteria(bool mc);
void set_samples(int s);
void set_seed(string s);
//generational maze experiments
typedef int (*successfunc)(population_state* ps);
typedef int (*epochfunc)(population_state* ps,int generation,successfunc sf);

Population *classifier_generational(char* output_dir,const char* classfile,int param, const char *genes, int gens, bool novelty); 
int classifier_success_processing(population_state* pstate);
int classifier_generational_epoch(population_state* pstate,int generation);

int generalized_generational_epoch(population_state* pstate,int generation,successfunc success_processing);
void destroy_organism(Organism* curorg);

//Walker novelty steady-state 
noveltyitem* classifier_novelty_map(Organism *org,data_record* record=NULL);

#endif
