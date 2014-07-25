// Multi-Objective Multi-Agent Rover Domain.
// For use with Difference Rewards in SPEA2 and NSGA-II
// Core code: Logan Yliniemi, 2012.
// NSGAheader, SPEAheader, NNLIBv2, Logan Yliniemi, 2013.
// Alterations: Logan Yliniemi and Drew T. Wilson, July 2014

#include <iostream>
#include <math.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include "NNV.h"
#include "NSGAheader.h"

#define pi 3.141529
#define QUADRANTS 4
#define XMIN 0
#define XMAX 100
#define YMIN 0
#define YMAX 100

#define num_POI 4
#define num_ROVERS 4
#define DETERMINISTICALLY_PLACED 0 // If more than num_ROVERS, will deterministcally place all rovers

#define DO_NSGA 1
#define TIMESTEPS 10
#define GENERATIONS 1

#define ROVERWATCH 0 // Index of rover to watch.
#define MIN_OBS_DIST (XMAX/100)

// NEURAL NETWORK PARAMETERS
#define INPUTS 12
#define HIDDEN 6
#define OUTPUTS 2
#define EVOPOP 5

using namespace std;

class landmark;
class rover;

void angle_resolve(double&);
void angle_resolve_pmpi(double&);
void xresolve(double&);
void yresolve(double&);
double find_distance(double, double, double, double);

void angle_resolve(double& angle)
{
	while (angle>2 * pi)
	{
		angle -= 2 * pi;
	}
	while (angle<0)
	{
		angle += 2 * pi;
	}
}

void angle_resolve_pmpi(double& angle)
{
	while (angle>pi)
	{
		angle -= 2 * pi;
	}
	while (angle<-pi)
	{
		angle += 2 * pi;
	}
}

void xresolve(double& x)
{
	while (x<XMIN)
	{
		x = XMIN + 1;
	}
	while (x>XMAX)
	{
		x = XMAX - 1;
	}
}

void yresolve(double& y)
{
	while (y<YMIN)
	{
		y = YMIN + 1;
	}
	while (y>YMAX)
	{
		y = YMAX - 1;
	}
}

double find_distance(double x, double y, double tarx, double tary)
{
	double dx = x - tarx;
	double dy = y - tary;

	double a = sqrt(dx*dx + dy*dy);

	return a;
}

class rover
{
public:
	double heading;
	double x;
	double y;
	double xstart;
	double ystart;
	double headingstart;
	double xdot;
	double ydot;
	int ID;
	double rover_state[QUADRANTS];
	double blue_state[QUADRANTS];
	double red_state[QUADRANTS];
	vector<double> sum_global_red, sum_global_blue;
	vector<double> global_chunks, global_red_chunks, global_blue_chunks;
	vector<double> difference_chunks;
	vector<double> perfectly_learnable_chunks;
        
        vector<neural_network> population;
        vector<int> selected;

	int basic_sensor(double, double, double, double, double);
	void reset();
	int place(double, double, double);
	void replace();
	double strength_sensor(double, double, double);
	void move();
	void full_red_sensor(landmark*);
	void full_blue_sensor(landmark*);
	void full_rover_sensor(vector<rover>&);


	double local_red;
	double local_blue;
        
        /// high-level functions
        void sense(landmark* POIs, vector<rover>& fidos);
        void decide(int);
        void act();
        void react(landmark* POIs);
        
};

class landmark
{
public:
	double red_value;
	double blue_value;
	double start_red;
	double start_blue;
	double min_obs_distance;
	double max_obs_distance;
	double x;
	double y;
	vector<double> distances;

	void create(double, double, double, double);
	void reset();


	int find_kth_closest_rover(int, vector<rover>&);
	double find_dist_to_rover(int, vector<rover>&);
	int find_kth_closest_rover_not_i(int, int, vector<rover>);
	void find_dist_to_all_rovers(vector<rover>&);

	double calc_red_observation_value(double);
	double calc_blue_observation_value(double);
};

void landmark::create(double xpos, double ypos, double red, double blue)
{
	x = xpos;
	y = ypos;
	red_value = red;
	blue_value = blue;
	start_red = red_value;
	start_blue = blue_value;
	min_obs_distance = MIN_OBS_DIST; /// LYLY ADJUSTABLE
	max_obs_distance = XMAX / 1; /// LYLY ADJUSTABLE
}

void landmark::reset()
{
	red_value = start_red;
	blue_value = start_blue;
}

int landmark::find_kth_closest_rover(int k, vector<rover>& fidos)
{
	int closest;
	double closest_distance;
	vector<double> distances;
	for (int b = 0; b<num_ROVERS; b++){
		double delx, dely;
		delx = fidos.at(b).x - x;
		dely = fidos.at(b).y - y;
		double dis = sqrt(delx*delx + dely*dely);
		distances.push_back(dis);
	}
	sort(distances.begin(), distances.end());
	closest_distance = distances.at(k);
	for (int b = 0; b<num_ROVERS; b++){
		double delx, dely;
		delx = fidos.at(b).x - x;
		dely = fidos.at(b).y - y;
		double dis = sqrt(delx*delx + dely*dely);
		if (dis == closest_distance){
			closest = b;
			break;
		}
	}
	return closest;
}

int landmark::find_kth_closest_rover_not_i(int k, int i, vector<rover> fidos){
	int closest;
	double closest_distance;
	vector<double> distances;
	for (int b = 0; b<num_ROVERS; b++){
		if (b == i){ continue; }
		double delx, dely;
		delx = fidos.at(b).x - x;
		dely = fidos.at(b).y - y;
		double dis = sqrt(delx*delx + dely*dely);
		distances.push_back(dis);
	}
	sort(distances.begin(), distances.end());
	closest_distance = distances.at(k);
	for (int b = 0; b<num_ROVERS; b++){
		if (b == i){ continue; }
		double delx, dely;
		delx = fidos.at(b).x - x;
		dely = fidos.at(b).y - y;
		double dis = sqrt(delx*delx + dely*dely);
		if (dis == closest_distance){
			closest = b;
			break;
		}
	}
	return closest;
}

double landmark::find_dist_to_rover(int rvr, vector<rover>& fidos)
{
	double delx, dely;
	delx = fidos.at(rvr).x - x;
	dely = fidos.at(rvr).y - y;
	double dis = sqrt(delx*delx + dely*dely);

	return dis;
}

void landmark::find_dist_to_all_rovers(vector<rover>& fidos)
{
	distances.clear();
	for (int i = 0; i < num_ROVERS; i++)
	{
		distances.push_back(find_dist_to_rover(i, fidos));
	}
}

double landmark::calc_red_observation_value(double d)
{
	double val;
	d = fmax(d, min_obs_distance);
	if (d>max_obs_distance)
	{
		return 0;
	}
	val = red_value / d;
        return val;
}

double landmark::calc_blue_observation_value(double d)
{
	double val;
	d = fmax(d, min_obs_distance);
	if (d>max_obs_distance)
	{
		return 0;
	}
	val = blue_value / d;
	return val;
}

void rover::replace()
{
	/// resets the rovers to starting position after each policy is implemented
	heading = headingstart;
	x = xstart;
	y = ystart;
	xdot = 0;
	ydot = 0;
	local_blue = 0;
	local_red = 0;
}

void rover::reset()
{
	/// clears the rover's information, for easier debugging.
	heading = 0;
	x = 0;
	y = 0;
	xdot = 0;
	ydot = 0;
	local_blue = 0;
	local_red = 0;
}

void rover::move()
{
	x += xdot;
	y += ydot;
	xresolve(x);
	yresolve(y);
	heading = atan2(ydot, xdot);
}

int rover::place(double xspot, double yspot, double head)
{
	/// places this rover in the world with the specified x,y,theta.
	static int num;
	ID = num;
	num++;
	x = xspot;
	y = yspot;
	xstart = xspot;
	ystart = yspot;
	headingstart = head;
	heading = head;
	xresolve(x);
	yresolve(y);
	angle_resolve(heading);

	if (x>XMIN && y>YMIN && x<XMAX && y<YMAX)
	{
		return 0;
	}
	else
	{
		cout << "rover::place error" << endl;
		return 1;
	}
}

int deterministic_and_random_place(vector<rover>& fidos)
{
	// pseudo-randomly place a number of rovers
	srand(1);
	double x, y, heading;
	for (int i = 0; i < DETERMINISTICALLY_PLACED; i++)
	{
		if (i == num_ROVERS)
			return 0;
		x = rand() % 101;
		y = rand() % 101;
		heading = rand() % 361 * pi / 180;
		cout << x << " " << y << " " << heading << endl;
		fidos.at(i).place(x, y, heading);
	}

	// randomly place the rest of the rovers
	int left_over = num_ROVERS - DETERMINISTICALLY_PLACED;
	srand(time(NULL));
	for (int j = left_over; j > 0; j--)
	{
		x = rand() % 101;
		y = rand() % 101;
		heading = rand() % 361 * pi / 180;
		cout << x << " " << y << " " << heading << endl;
		fidos.at(num_ROVERS-j).place(x, y, heading);
	}
	return 0;
}

int rover::basic_sensor(double roverx, double rovery, double rover_heading, double tarx, double tary)
{
	
	double dx;
	double dy;

	dx = tarx - roverx;
	dy = tary - rovery;

	// heading to target with respect to robot frame
	double tarheading;
	tarheading = atan2(dy, dx);

	double del_heading;
	del_heading = tarheading - rover_heading;
	angle_resolve_pmpi(del_heading);

	//cout << "del_heading: " << del_heading << endl;

	double nw = pi / 4;
	double ne = -pi / 4;
	double sw = 3 * pi / 4;
	double se = -3 * pi / 4;

	//cout << "Deltas (x,y) : " << dx << "\t" << dy << endl;

	if (del_heading<nw && del_heading>ne)
	{
		/// object is "ahead" of the robot.
		//cout << "Ahead" << endl;
		return 0;
	}
	if (del_heading >= nw && del_heading<sw)
	{
		//cout << "Left" << endl;
		/// object is "left" of the robot
		return 1;
	}
	if (del_heading <= ne && del_heading>se)
	{
		//cout << "Right" << endl;
		///object is "right" of the robot
		return 2;
	}
	if (del_heading <= se || del_heading >= sw)
	{
		//cout << "Behind" << endl;
		///object is "behind" the robot
		return 3;
	}

	else
	{
		cout << "problems in basic_sensor;" << endl;
		return 66;
	}
}

double rover::strength_sensor(double value, double tarx, double tary)
{
    double numerator;
    double denominator;
    double strength;
        
        numerator = value;
        double delx = x-tarx;
        double dely = y-tary;
        denominator = fmax(sqrt(delx*delx+dely*dely),MIN_OBS_DIST);
       
	strength = numerator / denominator;
        
        return strength;
}

void rover::full_red_sensor(landmark* POIs)
{
	int quadrant;
	for (int i = 0; i < QUADRANTS; i++)
	{
		red_state[i] = 0;
	}
	for (int i = 0; i<num_POI; i++)
	{
		quadrant = basic_sensor(x, y, heading, POIs[i].x, POIs[i].y);
		double value = POIs[i].red_value;
		double tarx = POIs[i].x;
		double tary = POIs[i].y;
		double str = strength_sensor(value, tarx, tary);

		red_state[quadrant] += str;
	}
//        if(ID == ROVERWATCH){
//            cout << "ROVER " << ROVERWATCH << " RED STATE" << endl;
//            for(int i=0; i<QUADRANTS; i++){
//                cout << red_state[i] << "\t";
//            }
//            cout << endl;
//        }
//	for (int i = 0; i < QUADRANTS; i++)
//	{
//		cout << red_state[i] << "\t";
//	}
//	cout << endl;
}

void rover::full_blue_sensor(landmark* POIs)
{
	int quadrant;
	for (int i = 0; i < QUADRANTS; i++)
	{
		blue_state[i] = 0;
	}
	for (int i = 0; i<num_POI; i++)
	{
		quadrant = basic_sensor(x, y, heading, POIs[i].x, POIs[i].y);
		double value = POIs[i].blue_value;
		double tarx = POIs[i].x;
		double tary = POIs[i].y;
		double str = strength_sensor(value, tarx, tary);

		blue_state[quadrant] += str;
	}
//        if(ID == ROVERWATCH){
//            cout << "ROVER " << ROVERWATCH << " BLUE STATE" << endl;
//            for(int i=0; i<QUADRANTS; i++){
//                cout << blue_state[i] << "\t";
//            }
//            cout << endl;
//        }
}

void rover::full_rover_sensor(vector<rover>& fidos)
{
	int quadrant;
	for (int i = 0; i < QUADRANTS; i++)
	{
		rover_state[i] = 0;
	}
	for (int r = 0; r<num_ROVERS; r++)
	{
		if (r == ID)
		{
			continue;
		}
		quadrant = basic_sensor(x, y, heading, fidos.at(r).x, fidos.at(r).y);
		double value = 1;
		double tarx = fidos.at(r).x;
		double tary = fidos.at(r).y;
		double str = strength_sensor(value, tarx, tary);

		rover_state[quadrant] += str;
	}
//        if(ID == ROVERWATCH){
//            cout << "ROVER " << ROVERWATCH << " ROVER STATE" << endl;
//            for(int i=0; i<QUADRANTS; i++){
//                cout << rover_state[i] << "\t";
//            }
//            cout << endl;
//       }
}

void clear_rewards(vector<rover>& fidos)
{
	for (int i = 0; i < num_ROVERS; i++)
	{
		fidos.at(i).global_chunks.clear();
		fidos.at(i).perfectly_learnable_chunks.clear();
		fidos.at(i).difference_chunks.clear();
	}
}

void basic_rover_sensor_testing(){
    // Stuff for testing

	rover testrover;
	landmark testlandmark;

	testrover.reset();
	testlandmark.reset();

	testrover.place(50, 50, 0);
	testlandmark.create(40, 40, 10, 10);

	double turning = 0;
	int q;

	///*
	while (turning < 2 * pi) {
		q = testrover.basic_sensor(testrover.x, testrover.y, testrover.heading, testlandmark.x, testlandmark.y);
		cout << "Landmark is in quadrant " << q << " with respect to the rover." << endl;
		testrover.heading += pi / 8;
		turning = testrover.heading;
	}
	//*/

	// End testing
}

void rover_place_test(vector<rover>& fidos)
{
	double x, y, heading;

	x = 50;// 10;
	y = 50;// 20;
	heading = 0;
	fidos.at(0).place(x, y, heading);

	x = 50;//10;
	y = 50;//80;
	heading = 0;
	fidos.at(1).place(x, y, heading);

	x = 50;//90;
	y = 50;//20;
	heading = 0;
	fidos.at(2).place(x, y, heading);

	x = 50;//90;
	y = 50;//80;
	heading = 0;
	fidos.at(3).place(x, y, heading);
}

FILE* open_files(){
    FILE* pFILE = fopen ("Testing.txt","w");
    return pFILE;
}

void print_rover_locations(FILE * pFILE1, vector<rover>& fidos)
{
	vector<double> temp_store;

	for (int i = 0; i < num_ROVERS; i++)
	{
		temp_store.push_back(fidos.at(i).x);
		temp_store.push_back(fidos.at(i).y);
	}

	for (int j = 0; j < num_ROVERS * 2; j++)
	{
		fprintf(pFILE1, "%.4f\t", temp_store.at(j));
	}

	fprintf(pFILE1, "\n");
}

void print_poi_locations(FILE * pFILE2, vector<double> x, vector<double> y)
{
	for (int i = 0; i < num_POI; i++)
	{
		fprintf(pFILE2, "%.2f\t%.2f\n", x.at(i), y.at(i));
	}
}

vector<double> kill_lowest_performers(vector<neural_network>* pNN, int r);
void expand_population(vector<neural_network>* pNN, int r, vector<double>);

void rover::sense(landmark* POIs, vector<rover>& fidos){
    //cout << "sensing " << r << endl;
    full_blue_sensor(POIs);
    full_red_sensor(POIs);
    full_rover_sensor(fidos);
    //cout << "end sensing " << r << endl;
}

void rover::decide(int ev){
    vector<double> inp;
    //inp.push_back(fidos[r].x);
    //inp.push_back(fidos[r].y);
    for (int i = 0; i<QUADRANTS; i++)
    {
            inp.push_back(red_state[i]);
    }
    for (int i = 0; i<QUADRANTS; i++)
    {
            inp.push_back(blue_state[i]);
    }
    for (int i = 0; i<QUADRANTS; i++)
    {
            inp.push_back(rover_state[i]);
    }
    
    population.at(selected.at(ev)).clean();
    population.at(selected.at(ev)).take_vector_input(inp);
    
    population.at(selected.at(ev)).execute(INPUTS,OUTPUTS);
    //NN[i].scaleoutputs();
    //}
    //for(int i=0; i<EVOPOP; i++)
    //{
    //cout << "MAXO 0: " << maxo.at(0) << endl;
    //cout << "MAXO 1: " << maxo.at(1) << endl;
    xdot = population.at(selected.at(ev)).give_output(0);
            //output[0];
    ydot = population.at(selected.at(ev)).give_output(1);
            //output[1];
    //cout << NN[r][selected[r][ev]].output[0] << endl;
    //cout << "FIDODX " << fidos[r].xdot << endl;
    //cout << "FIDODY " << fidos[r].ydot << endl;
    //}
}

void rover::act(){
    //cout << "acting " << r << endl;
    //cout << "fidos x " << x << "\t";
    move();
    //cout << "fidos y " << y << endl;
    //cout << "end acting " << r << endl;
}

void rover::react(landmark* POIs){
    /*
     Left inside of main for the time being.
     */
}

int main()
{
	cout << "Hello world!" << endl;
	srand(time(NULL));
	
	FILE * pFILE1 = fopen("rover_locations.txt", "w");
	FILE * pFILE2 = fopen("poi_locations.txt", "w");
	
        /// BGN Create Landmarks
	landmark POIs[num_POI];

	/// x, y, r, b;
	POIs[0].create(10, 10, 10, 10);
	POIs[1].create(10, 90, 0, 100);
	POIs[2].create(90, 10, 100, 0);
	POIs[3].create(90, 90, 100, 100);

	vector<double> poi_x_locations;
	vector<double> poi_y_locations;

	for (int j = 0; j < num_POI; j++)
	{
		poi_x_locations.push_back(POIs[j].x);
		poi_y_locations.push_back(POIs[j].y);
	}

	print_poi_locations(pFILE2, poi_x_locations, poi_y_locations);
        /// END Create Landmarks
        
        /// BGN Create Rovers
        vector<rover> fidos(num_ROVERS);
        //vector<rover>* pfidos = &fidos;
	/// x,y,h
	for (int r = 0; r<num_ROVERS; r++)
	{
                fidos.at(r).reset();
                fidos.at(r).population.clear();
                neural_network NN;
                
                /// Set up neural network
                    NN.clean();
                    NN.setup(INPUTS,HIDDEN,OUTPUTS);
                    for(int in=0; in<INPUTS; in++){
                if(in<4){
                    NN.take_in_min_max(0,15);
                }
                if(in>=4){
                    NN.take_in_min_max(0,15);
                }
            }
            for (int out = 0; out<OUTPUTS; out++){
                NN.take_out_min_max(-XMAX / 10,XMAX / 10);
            }
                    
                /// create population of neural networks.  
                for(int p=0; p<EVOPOP; p++){
                    fidos.at(r).population.push_back(NN);
                    fidos.at(r).selected.push_back(p);
                }
	}

	deterministic_and_random_place(fidos);
	//rover_place_test(fidos);
        /// END Create Rovers
        
	cout << "done with inputsoutputs scaling" << endl;
        
	NSGA_2 NSGA;
	NSGA.declare_NSGA_dimension(2);

	vector<double> sum_global;
	vector<double> sum_perfect;
	vector<double> sum_difference;

	cout << "Preliminaries completed" << endl;
	for (int gen = 0; gen<GENERATIONS; gen++)
	{
            cout << "Beginning Generation " << gen << endl;
                for (int r = 0; r<num_ROVERS; r++)
                {
                    int SWAPS = 100;
                    for (int i = 0; i<SWAPS; i++)
                    {
                            int p1 = rand() % EVOPOP;
                            int p2 = rand() % EVOPOP;
                            int holder;
                            
                            holder = fidos.at(r).selected.at(p1);
                            fidos.at(r).selected.at(p1) = fidos.at(r).selected.at(p2);
                            fidos.at(r).selected.at(p2) = holder;
                    }
                }
		
		for (int ev = 0; ev<EVOPOP; ev++)
		{
                        clear_rewards(fidos);

			for (int k = 0; k<num_ROVERS; k++)
			{
				fidos.at(k).replace();
			}

			for (int t = 0; t<TIMESTEPS; t++)
			{
				//if(t%100==0){
				//cout << "." << flush;}
                            
				/// SENSE
				//cout << "Sense!" << endl;		
				for (int r = 0; r<num_ROVERS; r++)
				{
									fidos.at(r).heading = 0;
                                    fidos.at(r).sense(POIs,fidos);
				}

				/// DECIDE
				// Run the neural network here.
				//cout << "Decide!" << endl;
				for (int r = 0; r<num_ROVERS; r++)
				{
                                    fidos.at(r).decide(ev);
				}

				/// ACT
				//cout << "ACT!" << endl;
				if (gen == GENERATIONS - 1){
					print_rover_locations(pFILE1, fidos);
				}

				for (int r = 0; r<num_ROVERS; r++)
				{
                                    fidos.at(r).act();
				}

				/// REACT
				//cout << "REACT!" << endl;
				double red_observation_value = 0, blue_observation_value = 0;
				double global_red = 0;
				double global_blue = 0;
				double global = 0;

				for (int i = 0; i<num_POI; i++)
				{
					//cout << "begin react " << i << endl;

					//int assignee = POIs[i].find_kth_closest_rover(0, fidos);
					//double distance = POIs[i].find_dist_to_rover(assignee, fidos);
					//double red_observation_value = POIs[i].calc_red_observation_value(distance);
					//double blue_observation_value = POIs[i].calc_blue_observation_value(distance);

					//fidos.at(assignee).local_red += red_observation_value;
					//fidos.at(assignee).local_blue += blue_observation_value;

					POIs[i].find_dist_to_all_rovers(fidos);

					for (int j = 0; j < num_ROVERS; j++)
					{
						red_observation_value += POIs[i].calc_red_observation_value(POIs[i].distances.at(j));
						blue_observation_value += POIs[i].calc_blue_observation_value(POIs[i].distances.at(j));
					}

					global += red_observation_value + blue_observation_value;
					global_red += red_observation_value;
					global_blue += blue_observation_value;

					//cout << "at distance " << distance << endl;
					//cout << "red value of " << red_observation_value << " assigned to rover " << assignee << endl;
					//cout << "blue value of " << blue_observation_value << " assigned to rover " << assignee << endl;
					//cout << "end react " << i << endl;

					//int assignee_not_i=POIs[i].find_kth_closest_rover_not_i(0,i,&fido);
					//double distance_not_i=POIs[i].find_dist_to_rover(assignee_not_i,&fido);
				}

				//uses the distances to each rover to find the perfectly learnable reward and difference rewards
				for (int i = 0; i < num_ROVERS; i++)
				{
					double P_i = 0;
					double counterfactual = global;
					for (int j = 0; j < num_POI; j++)
					{
						P_i += POIs[j].calc_red_observation_value(POIs[j].distances.at(i));
						P_i += POIs[j].calc_blue_observation_value(POIs[j].distances.at(i));
						if (POIs[j].distances.at(i) < POIs[j].max_obs_distance)
						{
							counterfactual -= POIs[j].calc_red_observation_value(POIs[j].distances.at(i));
							counterfactual -= POIs[j].calc_blue_observation_value(POIs[j].distances.at(i));
						}
					}
					fidos.at(i).global_chunks.push_back(global);
					fidos.at(i).global_red_chunks.push_back(global_red);
					fidos.at(i).global_blue_chunks.push_back(global_blue);
					fidos.at(i).perfectly_learnable_chunks.push_back(P_i);
					fidos.at(i).difference_chunks.push_back(counterfactual);
				}
			}

			/// END TIMESTEP LOOP

			//////
			for (int i = 0; i < num_ROVERS; i++)
			{
				fidos.at(i).sum_global_red.push_back(accumulate(fidos.at(i).global_red_chunks.begin(), fidos.at(i).global_red_chunks.end(), 0.0));
				fidos.at(i).sum_global_blue.push_back(accumulate(fidos.at(i).global_blue_chunks.begin(), fidos.at(i).global_blue_chunks.end(), 0.0));
				sum_global.push_back(accumulate(fidos.at(i).global_chunks.begin(), fidos.at(i).global_chunks.end(), 0.0));
				sum_perfect.push_back(accumulate(fidos.at(i).perfectly_learnable_chunks.begin(), fidos.at(i).perfectly_learnable_chunks.end(), 0.0));
				sum_difference.push_back(accumulate(fidos.at(i).difference_chunks.begin(), fidos.at(i).difference_chunks.end(), 0.0));
			}

			for (int r = 0; r<num_ROVERS; r++)
			{
			    //VVNN.at(r).at(selected[r][ev]).set_fitness(fidos[r].local_red + fidos[r].local_blue);
                            fidos.at(r).population.at(fidos.at(r).selected.at(ev)).set_fitness(sum_global.at(r));
                            //VVNN.at(r).at(selected[r][ev]).set_fitness(sum_global.at(r));
			    //cout << fidos[r].local_red << " " << fidos[r].local_blue << endl;
			}
			sum_global.clear();
			sum_perfect.clear();
			sum_difference.clear();
		}
		/// END EVOPOP LOOP

		if (DO_NSGA){
			NSGA.NSGA_reset();
			
			for (int r = 0; r < num_ROVERS; r++) {
				for (int ev = 0; ev < EVOPOP; ev++) {
					vector<double> afit;
					afit.push_back(fidos.at(r).sum_global_red.at(ev));
					afit.push_back(fidos.at(r).sum_global_blue.at(ev));
					NSGA.vector_input(afit, ev);
				}
				NSGA.execute();
				for (int ev = 0; ev < EVOPOP; ev++) {
					//fidos.at(r).population.at(fidos.at(r).selected.at(ev)).set_fitness(NSGA.NSGA_member_fitness(ev));
				}
			}
		}


		//cout << "This generation's best local fitness is: " << VVNN.at(0).at(selected[0][0]).get_fitness() << endl;
		/*
		for (int iii = 0; iii < EVOPOP; iii++){
			for (int rove = 0; rove < num_ROVERS; rove++){
                            int spot = fidos.at(rove).selected.at(iii);
                            cout << "!!!!!!!" << fidos.at(rove).population.at(spot).get_fitness() << endl;
                        }
		}
		*/
		for (int r = 0; r<num_ROVERS; r++)
		{
                    vector<neural_network>* pVNN = &fidos.at(r).population;
                    vector<double> fit = kill_lowest_performers(pVNN,r);
                    expand_population(pVNN,r,fit);
		}

		for (int r = 0; r < num_ROVERS; r++)
		{
			for (int i = 0; i < EVOPOP; i++)
			{
				fidos.at(r).population.at(i).clean();
			}
		}
                
                /// GENERATION COMPLETED
	}

	//*/
	return 0;
}



vector<double> kill_lowest_performers(vector<neural_network>* pNN, int r){
    //vector<int> kill;
    /// r is the rover population we are working with.
    /// We kill the 'n' lowest performing NNs (mark for replacement in expand_population)
    
    // first, assemble a vector of these neural network fitnesses.
    vector<double> fitnesses;
    for(int i=0; i<EVOPOP; i++){
        fitnesses.push_back(pNN->at(i).get_fitness());
    }
    
    /// for as many as we need to eliminate...
    for(int rep=0; rep<EVOPOP/2; rep++){
    /// find the lowest index, 1 at a time.
    double lowest_fitness = 99999999999999; /// high fitness;
    int lowest_dex = -1; /// dummy index;
    for(int i=0; i<fitnesses.size(); i++){
        double debug_1 = fitnesses.at(i);
        double debug_2 = fitnesses.size();
        if(fitnesses.at(i) < lowest_fitness){
            lowest_fitness = fitnesses.at(i);
            lowest_dex = i;
        }
    }
    /// kill lowest fitness.
    fitnesses.erase(fitnesses.begin()+lowest_dex);
    /// kill matching population member.
    pNN->erase(pNN->begin()+lowest_dex);
    //kill.push_back(lowest_fitness);
    }
    
//    return kill;
    return fitnesses;
}

void expand_population(vector<neural_network>* pNN, int r, vector<double> fitnesses){
    //vector<int> survivors;
    /// create new neural networks like the survivors.
    for(int i=0; i<EVOPOP/2; i++){
        int spot; // the index of the one we're replicating.
        if (LYRAND < 0.8) {/// THIS ONE SELECTS THE BEST TO REPLICATE
                spot = max_element(fitnesses.begin(), fitnesses.end()) - fitnesses.begin();
            } else {
                /// THIS ONE SELECTS A RANDOM SURVIVOR TO REPLICATE.
                spot = rand() % pNN->size();
            }
        pNN->push_back(pNN->at(spot));
        pNN->back().mutate();
    }
}