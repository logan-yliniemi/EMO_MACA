
/* 
 * File:   NNV.h
 * Author: ylinieml
 *
 * Created on June 19, 2013, 10:29 AM
 */

#ifndef NNV_H
#define	NNV_H

#include <vector.h>
#include <math.h>
#include <map.h>

#define LYRAND ((double)rand()/RAND_MAX)
//#define BEGIN (sqrt(5))
#define BEGIN 0.01



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




//#define WEIGHTSTEP (0.2)
double WEIGHTSTEP;

int ctr_1=0;



class neural_network;
class layer;
class node;

class node{
    friend class layer;
    friend class neural_network;
    //vector<double> ins;
    //vector<double> inweights;
    double min;
    double max;
    //void scale_inputs();
    //void scale_outputs();
    void sigmoid();
    
    double inp;
    double out;
    vector<int> outconnections;
    vector<double> outweights;
    void clean();
    void setup(int,int);
    void make_full_connected(int);
    void alter_some_weights(double);
    
    void output_floor();
    void output_ceiling();
    
    bool bias;
    
public:
    void zero_weights();
    double get_average_outweights();
};

double node::get_average_outweights(){
    double num;
    for(int i=0; i<outweights.size(); i++){
        num+=fabs(outweights.at(i));
    }
    num/=outweights.size();
    return num;
}

void node::zero_weights(){
    for(int i=0; i<outweights.size(); i++){
        outweights.at(i)=0;
    }
}

void node::make_full_connected(int num){
    for(int i=0; i<num; i++){
    outconnections.push_back(i);
    outweights.push_back(LYRAND*BEGIN-LYRAND*BEGIN);
    }
}
void node::alter_some_weights(double odds){
    int I=outweights.size();
    for(int i=0; i<I; i++){
        double roll=LYRAND;
        if(roll<odds){
        outweights.at(rand()%I)=outweights.at(rand()%I) +LYRAND*WEIGHTSTEP -LYRAND*WEIGHTSTEP;
        }
    }
}

void node::clean(){
    inp=0;
    out=0;
}
void node::setup(int mi, int ma){
    clean();
    min=mi;
    max=ma;
    //outconnections.clear();
    //outweights.clear();
}

class layer{
    friend class node;
    friend class neural_network;
    vector<node> layer_nodes;
    void clean();
};

class neural_network{
    friend class node;
    friend class layer;
    
    vector<double> input_values;
    vector<double> input_minimums;
    vector<double> input_maximums;
    
    vector<double> output_values;
    vector<double> output_minimums;
    vector<double> output_maximums;
    
    layer input;
    layer hidden;
    layer output;
    
    
    void set_node_scaling(node*,bool,bool,int);
    
    double fitness;
    
public:
    void clean();
    void execute(int,int);
    void setup(int,int,int);
    void take_input(double);
    void take_vector_input(vector<double>);
    void take_in_min_max(double,double);
    void take_out_min_max(double,double);
    double give_output(int);
    void disp_outputs();
    void set_fitness(double);
    double get_fitness();
    void mutate();
    void zero_weights();
    void display_out_min_max(int);
    double get_average_weights();
};

double neural_network::get_average_weights(){
    double num=0;
   for(int i=0; i<input.layer_nodes.size(); i++){
        num+=input.layer_nodes.at(i).get_average_outweights();
    }
    for(int h=0; h<hidden.layer_nodes.size(); h++){
        num+=hidden.layer_nodes.at(h).get_average_outweights();
    }
    num/=((input.layer_nodes.size()+1)*hidden.layer_nodes.size() + (hidden.layer_nodes.size()+1)*output.layer_nodes.size());
    //cout << "AVERAGE NETWORK WEIGHT IS: " << num << endl;
    return num;
}

void neural_network::zero_weights(){
    for(int i=0; i<input.layer_nodes.size(); i++){
        input.layer_nodes.at(i).zero_weights();
    }
    for(int h=0; h<hidden.layer_nodes.size(); h++){
        hidden.layer_nodes.at(h).zero_weights();
    }
}

void neural_network::mutate(){
    if(rand()%2){
    for(int i=0; i<input.layer_nodes.size(); i++){
        input.layer_nodes.at(i).alter_some_weights(0.2);
    }
    for(int h=0; h<hidden.layer_nodes.size(); h++){
        hidden.layer_nodes.at(h).alter_some_weights(0.2);
    }
    }
    else{
       for(int i=0; i<input.layer_nodes.size(); i++){
        input.layer_nodes.at(i).alter_some_weights(0.8);
    }
    for(int h=0; h<hidden.layer_nodes.size(); h++){
        hidden.layer_nodes.at(h).alter_some_weights(0.8);
    } 
    }
}

void neural_network::set_fitness(double a){
    fitness=a;
}
double neural_network::get_fitness(){
    return fitness;
}

void neural_network::disp_outputs()
{
    cout << "DISP " << output_values.size() << ":";
    for(int i=0; i<output_values.size(); i++)
    {
        cout << output_values.at(i) << "\t";
    }
    cout << endl;
}

void neural_network::take_in_min_max(double a, double b){
    input_minimums.push_back(a);
    input_maximums.push_back(b);
}

void neural_network::take_out_min_max(double a, double b){
    output_minimums.push_back(a);
    output_maximums.push_back(b);
}
void neural_network::display_out_min_max(int output_node){
    cout << output_minimums.at(output_node) << "\t" << output_maximums.at(output_node) << endl;
}

void neural_network::take_input(double a){
    ctr_1++;
    input_values.push_back(a);
}

void neural_network::take_vector_input(vector<double> a){
    input_values.clear();
    input_values = a;
}

double neural_network::give_output(int spot){
    return output_values.at(spot);
}

void layer::clean(){
    for(int i=0; i<layer_nodes.size(); i++){
        layer_nodes.at(i).clean();
    }
}

void neural_network::clean(){
    input_values.clear();
    output_values.clear();
    input.clean();
    hidden.clean();
    output.clean();
    fitness=-14;
}


//void node::scale_inputs(){
//    inp=inp-min;
//    inp=inp/max;
//}

//void node::scale_outputs(){
//    out=out*max;
//    out=out+min;
//}

void node::sigmoid(){
    //
    //map<int,double>::iterator it;
    
    //out = 1/(1+exp(-inp));
    //return;
    
    static map< int, double > sig_iomap;
    int input_int = inp*TRIG_GRANULARITY;
    if(sig_iomap.count(input_int) == 1)
    {out = sig_iomap.find(input_int)->second;}
    else{
      out = 1/(1+exp(-inp));
      sig_iomap.insert(pair <int,double> (input_int,out));
    }
//    
//    static int counter;
//    static double mini;
//    static double maxi;
//    
//    //cout << "MINI: " << mini << endl;
//    
//    if(counter==0){
//        mini=9999999;
//        maxi=-999999;
//    }
//    
//    if(inp < mini){
//        mini=inp;
//    }
//    if(inp > maxi){
//        maxi=inp;
//    }
//    
//    if(counter%1000000==0){
//        
//        cout << "\t\t\t\t\t\t MAP SIZE: " << sig_iomap.size() << "\t" << mini << "\t" << maxi << endl;
//    }
//    
//    counter++;
    
    //static int counter;
    //static double error;
    //error+= fabs(out - 1/(1+exp(-inp)));
    //counter++;
    //cout << (double)error/counter << endl;
}

void neural_network::setup(int inp, int hid, int out){
    /// clean up for initials.
    input.layer_nodes.clear();
    hidden.layer_nodes.clear();
    output.layer_nodes.clear();
    
    for(int i=0; i<inp+1; i++){
    node N;
    //node* pN=&N;
    N.clean();
    N.bias=false;
    N.make_full_connected(hid);
    if(i==inp){N.bias=true;}
    //if(N.bias==false){set_node_scaling(pN,true,false,i);}
    input.layer_nodes.push_back(N);
    }
    for(int h=0; h<hid+1; h++){
    node N;
    N.clean();
    N.bias=false;
    N.make_full_connected(out);
    if(h==hid){N.bias=true;}
    hidden.layer_nodes.push_back(N);
    }
    for(int o=0; o<out; o++){
    node N;
    //node* pN=&N;
    N.clean();
    //if(N.bias==false){set_node_scaling(pN,false,true,o);}
    output.layer_nodes.push_back(N);
    }
    
}

//void neural_network::set_node_scaling(node* pN, bool ip, bool op, int node_num){
//    
//    if(ip==true){
//        pN->min=input_minimums.at(node_num);
//        pN->max=input_maximums.at(node_num);
//    }
//    if(op==true){
//        pN->min=output_minimums.at(node_num);
//        pN->max=output_maximums.at(node_num);
//    }
//}

void neural_network::execute(int input_number, int output_number){
    int hidden_number=hidden.layer_nodes.size()-1;
    //input_number++; /// allow for bias
    
    /// input takes input;

    for(int i=0; i<input_number; i++){
        input.layer_nodes.at(i).inp=
                (input_values.at(i)-input_minimums.at(i))/(input_maximums.at(i)-input_minimums.at(i));
        //input.layer_nodes.at(i).scale_inputs();
        input.layer_nodes.at(i).sigmoid();
        //cout << "INPUT LAYER: " << input.layer_nodes.at(i).out<< endl;
    }
    input.layer_nodes.back().out=1;
    
//    if(rand()%10000==0)
//    {
//        cout << "NN INPUT SAMPLES" << endl;
//        for(int i=0; i<input_number; i++){
//        cout << input_values.at(i) << "\t";
//        }
//        cout << endl;
//        for(int i=0; i<input_number; i++){
//            cout << input.layer_nodes.at(i).inp << "\t";
//        }
//        cout << endl;
//    }
    
    //cout << input.layer_nodes.size() << endl;
    //cout << "INPUT LAYERBIAS : " << input.layer_nodes.back().out<< endl;
    /// input to hidden;
    for(int i=0; i<input_number+1; i++){
        for(int j=0; j<input.layer_nodes.at(i).outconnections.size(); j++){
            /// value to get output to input.layer_nodes.at(i).outconnections.at(j)
            double value = input.layer_nodes.at(i).outweights.at(j)*input.layer_nodes.at(i).out;
            /// spot to put out to
            int to = input.layer_nodes.at(i).outconnections.at(j);
            /// execute it.
            hidden.layer_nodes.at(to).inp+=value;
        }
    }
    
    for(int h=0; h<hidden_number; h++){
        hidden.layer_nodes.at(h).sigmoid();
        //cout << "H LAYER: " << hidden.layer_nodes.at(h).out<< endl;
    }
    hidden.layer_nodes.back().out=1;
     //cout << "H LAYERBIAS : " << hidden.layer_nodes.back().out<< endl;
    
    /// hidden to output;
    for(int h=0; h<hidden_number; h++){
        for(int j=0; j<hidden.layer_nodes.at(h).outconnections.size(); j++){
            double value = hidden.layer_nodes.at(h).outweights.at(j)*hidden.layer_nodes.at(h).out;
            int to = hidden.layer_nodes.at(h).outconnections.at(j);
            //cout << "HIDDEN VALUE: " << value << "\t TO " << to << endl;
            output.layer_nodes.at(to).inp+=value;
        }
    }
    
    /// output nodes to output values;
    output_values.clear();
    for(int o=0; o<output_number; o++){
        output.layer_nodes.at(o).sigmoid();
        //cout << "O LAYER: " << o << "::" << output.layer_nodes.at(o).out << endl;
        output.layer_nodes.at(o).output_floor();
        //cout << "O LAYERfloor: " << o << "::" << output.layer_nodes.at(o).out<< endl;
        output.layer_nodes.at(o).output_ceiling();
        //cout << "O LAYERceil: " << o << "::" << output.layer_nodes.at(o).out<< endl;
        //output.layer_nodes.at(o).scale_outputs();
        output_values.push_back(output.layer_nodes.at(o).out*(output_maximums.at(o)-output_minimums.at(o))+output_minimums.at(o));
        //cout << "O LAYERout: " << output.layer_nodes.at(o).out*(output_maximums.at(o)-output_minimums.at(o))+output_minimums.at(o) << endl;
    }
}

void node::output_floor(){
        if(out<0.05){
            out=0;
        }
}
void node::output_ceiling(){
    if(out>0.95){
        out=1;
    }
}
#endif	/* NNV_H */