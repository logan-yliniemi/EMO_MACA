

#ifndef GEA_H
#define	GEA_H

#include <vector>
#include <math.h>
#include <map>
#include <random>

#define LYRAND ((double)rand()/RAND_MAX)

using namespace std;

/// BGN NON NNV, BUT NECESSARY FUNCTION BLOC
#define TRIG_GRANULARITY 1000
double sine(double inp){
    //return sin(inp);
    
    double out;
    static map< int, double > sine_iomap;
    int input_int = inp*TRIG_GRANULARITY;
    if(sine_iomap.count(input_int) == 1)
    {out = sine_iomap.find(input_int)->second;}
    else{
      out = sin(inp);
      sine_iomap.insert(pair <int,double> (input_int,out));
    }
    return out;
}
double cosine(double inp){
    //return cos(inp);
    
    double out;
    static map< int, double > sine_iomap;
    int input_int = inp*TRIG_GRANULARITY;
    if(sine_iomap.count(input_int) == 1)
    {out = sine_iomap.find(input_int)->second;}
    else{
      out = cos(inp);
      sine_iomap.insert(pair <int,double> (input_int,out));
    }
    return out;
}
/// END NON NNV, NECESSARY FUNCTION BLOC

#define WEIGHTSTEP (0.2)
//double WEIGHTSTEP;

int ctr_1=0;



class gaussian_evo_agent;

class gaussian_evo_agent{
    
    int num_waypoints;
    vector< vector<double> > waypoints;
    vector<double> variance;

    vector<double> input_values;
    vector<double> input_minimums;
    vector<double> input_maximums;
    
    vector<double> output_values;
    vector<double> output_minimums;
    vector<double> output_maximums;
    
    double fitness;
    vector<double> raw_objectives;
    vector<double> raw_locals;
    vector<double> raw_globals;
    vector<double> raw_gzmis;
    vector<double> raw_differences;
    
public:
	void clean();
    void execute();
    void setup();
    void take_input(double);
    void take_vector_input(vector<double>);
    void take_in_min_max(double,double);
    void take_out_min_max(double,double);
    double give_output(int);
    void disp_outputs();
    void set_fitness(double);
    double get_fitness();
    void mutate();
    void display_out_min_max(int);
    
    /// Objective and Reward Handling Functions
    double get_raw_objective(int);
    double get_raw_local(int);
    double get_raw_global(int);
    double get_raw_gzmi(int);
    double get_raw_difference(int);
    void set_next_raw_objective(double);
    void set_next_raw_local(double);
    void set_next_raw_global(double);
    void set_next_raw_gzmi(double);
    void set_next_raw_difference(double);
    void clear_raw_local();
    void clear_raw_global();
    void clear_raw_gzmi();
    void clear_raw_difference();
    void clear_raw_objectives();
    vector<double> get_raw_objectives();
    vector<double> get_raw_gzmis();
};

vector<double> gaussian_evo_agent::get_raw_objectives(){
    return raw_objectives;
}

vector<double> gaussian_evo_agent::get_raw_gzmis(){
    return raw_gzmis;
}

double gaussian_evo_agent::get_raw_objective(int a){
    return raw_objectives.at(a);
}

double gaussian_evo_agent::get_raw_local(int a){
    return raw_locals.at(a);
}

double gaussian_evo_agent::get_raw_global(int a){
    return raw_globals.at(a);
}

double gaussian_evo_agent::get_raw_gzmi(int a){
    return raw_gzmis.at(a);
}

double gaussian_evo_agent::get_raw_difference(int a){
    return raw_differences.at(a);
}

void gaussian_evo_agent::set_next_raw_objective(double b){
    raw_objectives.push_back(b);
}

void gaussian_evo_agent::set_next_raw_local(double b){
    raw_locals.push_back(b);
}

void gaussian_evo_agent::set_next_raw_global(double b){
    raw_globals.push_back(b);
}
void gaussian_evo_agent::set_next_raw_gzmi(double b){
    raw_gzmis.push_back(b);
}
void gaussian_evo_agent::set_next_raw_difference(double b){
    raw_differences.push_back(b);
}

void gaussian_evo_agent::clear_raw_local(){
    raw_locals.clear();
}

void gaussian_evo_agent::clear_raw_global(){
    raw_globals.clear();
}

void gaussian_evo_agent::clear_raw_gzmi(){
    raw_gzmis.clear();
}

void gaussian_evo_agent::clear_raw_difference(){
    raw_differences.clear();
}

void gaussian_evo_agent::clear_raw_objectives(){
    raw_objectives.clear();
}

void gaussian_evo_agent::mutate(){
    for(int i=0; i<waypoints.size(); i++){
        for(int j=0; j<waypoints.at(i).size(); j++){
            if(rand()%2){
                waypoints.at(i).at(j) += LYRAND*10;
                waypoints.at(i).at(j) -= LYRAND*10;
            }
        }
    }
}

void gaussian_evo_agent::set_fitness(double a){
    fitness=a;
}
double gaussian_evo_agent::get_fitness(){
    return fitness;
}

void gaussian_evo_agent::disp_outputs()
{
    cout << "DISP " << output_values.size() << ":";
    for(int i=0; i<output_values.size(); i++)
    {
        cout << output_values.at(i) << "\t";
    }
    cout << endl;
}

void gaussian_evo_agent::take_in_min_max(double a, double b){
    input_minimums.push_back(a);
    input_maximums.push_back(b);
}

void gaussian_evo_agent::take_out_min_max(double a, double b){
    output_minimums.push_back(a);
    output_maximums.push_back(b);
}
void gaussian_evo_agent::display_out_min_max(int output_node){
    cout << output_minimums.at(output_node) << "\t" << output_maximums.at(output_node) << endl;
}

void gaussian_evo_agent::take_input(double a){
    ctr_1++;
    input_values.push_back(a);
}

void gaussian_evo_agent::take_vector_input(vector<double> a){
    input_values.clear();
    input_values = a;
}

double gaussian_evo_agent::give_output(int spot){
    return output_values.at(spot);
}

void gaussian_evo_agent::clean(){
    fitness=0;
    raw_objectives.clear();
    raw_locals.clear();
    raw_globals.clear();
    raw_gzmis.clear();
    raw_differences.clear();
    
    input_values.clear();
    output_values.clear();
}

void gaussian_evo_agent::setup(){
    num_waypoints=1;
    waypoints.clear();
    variance.clear();
    vector<double> xy;
    xy.push_back(LYRAND*XMAX);
    xy.push_back(LYRAND*YMAX);
    waypoints.push_back(xy);
    variance.push_back(0.25);
    variance.push_back(0.25);
}

void gaussian_evo_agent::execute(){
    double x;
    double y;
    
    double stdevx = sqrt(variance.at(0));
    double stdevy = sqrt(variance.at(1));
    
    x = waypoints.at(0).at(0);
    y = waypoints.at(0).at(1);
    
    std::default_random_engine generator;
    std::normal_distribution<double> xdistribution(x,stdevx);
    std::normal_distribution<double> ydistribution(y,stdevy);
    
    double xn = xdistribution(generator);
    double yn = ydistribution(generator);
    
    //cout << "X,Y Coordinates: " << xn << " , " << yn << endl;
    
    output_values.push_back(xn);
    output_values.push_back(yn);
}
            
            
#endif	/* GEA_H */