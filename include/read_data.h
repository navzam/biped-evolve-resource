#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <vector>
using namespace std;

vector<vector<float> > read_classifier_data(const char* fname) {
  char input[1000];
  ifstream* file=new ifstream(fname);

  vector< vector<float> > data;

  while(!file->eof()) {
    file->getline(input,sizeof(input));
    if(strlen(input)==0)
      continue;
    vector<float> line_data;
    istringstream linestream(input);
    float next;
    while(!linestream.eof()) {
      linestream >> next;
      line_data.push_back(next);
    }
    data.push_back(line_data);
  }

  return data;
}
/*
int main(int argc, char** argv) {
vector<vector<float> > data = read_classifier_data("aust_nn.dat");
return 0;
}
*/
