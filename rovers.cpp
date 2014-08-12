// Multi-Objective Multi-Agent Rover Domain.
// For use with Difference Rewards in SPEA2 and NSGA-II
// Core code: Logan Yliniemi, 2012.
// NSGAheader, SPEAheader, NNLIBv2, Logan Yliniemi, 2013.
// Alterations: Logan Yliniemi and Drew T. Wilson, July 2014

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include "NNV.h"
#include "NSGAheader.h"
//#include "SPEAheader.h"

#define pi 3.141529
#define QUADRANTS 4
#define XMIN 0
#define XMAX 500
#define YMIN 0
#define YMAX 500

#define num_POI 200
#define num_ROVERS 10
#define DETERMINISTICALLY_PLACED 10

#define TELEPORTATION 1

#define DO_LOCAL 0
#define DO_GLOBAL 0
#define DO_DIFFERENCE 1
#define ALWAYS 1 /// For pieces that need to run regardless of options.

#define DO_LC 1
#define DO_HV 0
//#define DO_NSGA 0 /// broken into centralized, distributed.
//#define DO_SPEA 0 /// broken into centralized, distributed.

/// To replace do_nsga and do_spea.
#define DO_CENTRALIZED_NSGA 1
#define DO_CENTRALIZED_SPEA 0
#define DO_DISTRIBUTED_NSGA 0
#define DO_DISTRIBUTED_SPEA 0
#define DO_D_OF_NSGA_DISTRIBUTED 0 /// Requires DO_DIFFERENCE = 1

#define TIMESTEPS 1
#define GENERATIONS 1000
#define STAT_RUN 10

#define FITNESS_FILE_WATCH 0
#define ROVERWATCH 0
#define ROVERWATCHDEX 0 // Index of rover to watch.
#define MIN_OBS_DIST 1
#define MAX_OBS_DIST 3

// NEURAL NETWORK PARAMETERS
#define INPUTS 0
#define HIDDEN 5
#define OUTPUTS 2
#define EVOPOP 100

#define POI_GENERATE 0

using namespace std;

class landmark;
class rover;

void angle_resolve(double&);
void angle_resolve_pmpi(double&);
void xresolve(double&);
void yresolve(double&);
double find_distance(double, double, double, double);
void complete_react(vector<rover>&, landmark*);
void calculate_locals(vector<rover>& fidos, landmark* POIs);
void calculate_globals(vector<rover>& fidos, landmark* POIs);
void calculate_differences(vector<rover>& fidos, landmark* POIs);
void collect(vector<rover>& fidos, landmark* POIs, int ev);
vector<double> kill_lowest_performers(vector<neural_network>* pNN, int r, vector<rover>& fidos);
void expand_population(vector<neural_network>* pNN, int r, vector<double>);

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
	vector<double> local_red_chunks, local_blue_chunks;
	vector<double> global_red_chunks, global_blue_chunks;
	vector<double> difference_red_chunks, difference_blue_chunks;
    //vector<double> sum_local_red, sum_local_blue;
	//vector<double> sum_global_red, sum_global_blue;
	//vector<double> sum_difference_red, sum_difference_blue;
	vector<double> store_x, store_y;
	vector< vector<double> > policy_positions_x, policy_positions_y;

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
	// void react(landmark* POIs); // used as a general function, not a class method.

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
	int find_kth_closest_rover_not_i(int, int, vector<rover>&);
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
	max_obs_distance = MAX_OBS_DIST; /// LYLY ADJUSTABLE
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

int landmark::find_kth_closest_rover_not_i(int k, int i, vector<rover>& fidos){
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
    if(TELEPORTATION==0){
	x += xdot;
	y += ydot;
	xresolve(x);
	yresolve(y);
    heading = atan2(ydot, xdot);
    }
    if(TELEPORTATION==1){
        x=xdot;
        y=ydot;
        heading=0;
        if(ROVERWATCH && ID==ROVERWATCHDEX){
        cout << x << "\t" << y << "\t";
            // cout << endl;
        }
    }
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
	double x, y, heading;
	vector<double> xlist = { 40, 50, 50, 60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50 };
	vector<double> ylist = { 60, 40, 60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50 };
	vector<double> hlist = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	if (DETERMINISTICALLY_PLACED>xlist.size()){
		cout << "DETERMINISTIC PLACE ERROR" << endl;
	}

	for (int i = 0; i < DETERMINISTICALLY_PLACED; i++)
	{
		if (i == num_ROVERS)
			return 0;
		x = xlist.at(i);
		y = ylist.at(i);
		heading = hlist.at(i);
		//x = rand() % 101;
		//y = rand() % 101;
		//heading = rand() % 361 * pi / 180;
		cout << x << " " << y << " " << heading << endl;
		fidos.at(i).place(x, y, heading);
	}

	// randomly place the rest of the rovers
	int left_over = num_ROVERS - DETERMINISTICALLY_PLACED;
	//srand(time(NULL));
	for (int j = left_over; j > 0; j--)
	{
		x = rand() % 101;
		y = rand() % 101;
		heading = rand() % 361 * pi / 180;
		cout << x << " " << y << " " << heading << endl;
		fidos.at(num_ROVERS - j).place(x, y, heading);
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
	double delx = x - tarx;
	double dely = y - tary;
	denominator = fmax(sqrt(delx*delx + dely*dely), MIN_OBS_DIST);

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
	//        if(ROVERWATCH && ID == ROVERWATCHDEX){
	//            cout << "ROVER " << ROVERWATCHDEX << " RED STATE" << endl;
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
	//        if(ROVERWATCH && ID == ROVERWATCHDEX){
	//            cout << "ROVER " << ROVERWATCHDEX << " BLUE STATE" << endl;
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
	//        if(ROVERWATCH && ID == ROVERWATCHDEX){
	//            cout << "ROVER " << ROVERWATCHDEX << " ROVER STATE" << endl;
	//            for(int i=0; i<QUADRANTS; i++){
	//                cout << rover_state[i] << "\t";
	//            }
	//            cout << endl;
	//       }
}

void store_timestep_locations(vector<rover>& fidos)
{
	for (int i = 0; i < num_ROVERS; i++)
	{
		fidos.at(i).store_x.push_back(fidos.at(i).x);
		fidos.at(i).store_y.push_back(fidos.at(i).y);
	}
}

void store_policy_locations(vector<rover>& fidos)
{
	for (int i = 0; i < num_ROVERS; i++)
	{
		fidos.at(i).policy_positions_x.push_back(fidos.at(i).store_x);
		fidos.at(i).policy_positions_y.push_back(fidos.at(i).store_y);
	}
}

void clear_rewards(vector<rover>& fidos)
{
	for (int i = 0; i < num_ROVERS; i++)
	{
		fidos.at(i).local_red_chunks.clear();
		fidos.at(i).local_blue_chunks.clear();
		fidos.at(i).global_red_chunks.clear();
		fidos.at(i).global_blue_chunks.clear();
		fidos.at(i).difference_red_chunks.clear();
		fidos.at(i).difference_blue_chunks.clear();
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
	FILE* pFILE = fopen("Testing.txt", "w");
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
	static int once = 1;
	if (once == 1){
		for (int i = 0; i < num_POI; i++)
		{
			fprintf(pFILE2, "%.4f\t%.4f\n", x.at(i), y.at(i));
		}
	}
	once++;
}

void print_poi_all_values(FILE * pFILE5, vector<double> x, vector<double> y, vector<double> red, vector<double> blue)
{
	static int only_once;
	if (only_once == 0){
        cout << "PUTTING POIs IN FILE!!" << endl;
		for (int i = 0; i < num_POI; i++)
		{
			fprintf(pFILE5, "%.4f\t%.4f\t%.4f\t%.4f\n",
                    x.at(i),
                    y.at(i),
                    red.at(i),
                    blue.at(i));
		}
	}
	only_once++;
}

void print_fitnesses(FILE *pFILE3, vector<rover>& fidos, int gen)
{
	fprintf(pFILE3, "%s\t%d\n", "!!!!!!!!!!!!!!!!!!!!!GEN", gen);
	for (int iii = 0; iii < EVOPOP; iii++){
		for (int rove = 0; rove < num_ROVERS; rove++){
			int spot = fidos.at(rove).selected.at(iii);
			fprintf(pFILE3, "%.4f\n", fidos.at(rove).population.at(spot).get_fitness());
			//cout << fidos.at(rove).population.at(spot).get_fitness() << endl;
		}
		fprintf(pFILE3, "%s\t%i\n", "Evopop", iii);
	}
}

void print_red_blue_statrun(FILE * pFILE4, vector<rover>& fidos, int stat_run)
{
    //cout << "RBS: " << fidos.at(0).population.at(0).get_raw_objective(0) << "\t" << fidos.at(0).population.at(0).get_raw_objective(1) << endl;
	int i = 0;
	for (int ev = 0; ev < EVOPOP; ev++){
		fprintf(pFILE4, "%.5f\t", fidos.at(i).population.at(ev).get_raw_global(0));
		fprintf(pFILE4, "%.5f\t", fidos.at(i).population.at(ev).get_raw_global(1));
		fprintf(pFILE4, "%d\n", stat_run+1);
	}
}

void rover::sense(landmark* POIs, vector<rover>& fidos){
    if(TELEPORTATION==0){
	//cout << "sensing " << r << endl;
	full_blue_sensor(POIs);
	full_red_sensor(POIs);
	full_rover_sensor(fidos);
	//cout << "end sensing " << r << endl;
    }
    if(TELEPORTATION==1){
        /// stateless.
    }
}

void rover::decide(int ev){
    if(TELEPORTATION==0){
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
	population.at(selected.at(ev)).execute(INPUTS, OUTPUTS);
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
    if(TELEPORTATION==1){
        population.at(selected.at(ev)).clean();
        population.at(selected.at(ev)).execute(0,OUTPUTS);
        xdot = population.at(selected.at(ev)).give_output(0);
        ydot = population.at(selected.at(ev)).give_output(1);
    }
}

void rover::act(){
	//cout << "acting " << r << endl;
	//cout << "fidos x " << x << "\t";
	move();
	//cout << "fidos y " << y << endl;
	//cout << "end acting " << r << endl;
}

void complete_react(vector<rover>& fidos, landmark* POIs){
    bool globalflag=false;
    /// Find distance from each POI to each rover.
    for (int i = 0; i < num_POI; i++)
    {
        POIs[i].find_dist_to_all_rovers(fidos);
    }
    /// Find Local Rewards
    if(DO_LOCAL){
        calculate_locals(fidos, POIs);
    }
    /// Find Global Rewards
    if(DO_LOCAL || DO_GLOBAL || DO_DIFFERENCE){
        globalflag = true;
    calculate_globals(fidos, POIs); /// Always done so we can evaluate team performance.
    }
    /// Find Difference Rewards
    if(DO_DIFFERENCE) {
        calculate_differences(fidos, POIs);
    }
    /// Assign Rewards to Rovers (See Collect)
    
    /// Make sure global was calculated.
    if(globalflag==false){
        throw std::invalid_argument( "Didn't calculate globals." );
    }
    
}

void calculate_locals(vector<rover>& fidos, landmark* POIs){
    /// Each rover calculates its observation values for each POI, disregarding all other rovers.
    for (int i = 0; i < num_ROVERS; i++){
	for (int p = 0; p < num_POI; p++){
		int lred = POIs[p].calc_red_observation_value(POIs[p].distances.at(i));
		int lblue = POIs[p].calc_blue_observation_value(POIs[p].distances.at(i));
		fidos.at(i).local_red_chunks.push_back(lred);
		fidos.at(i).local_blue_chunks.push_back(lblue);
	}
    }
}

void calculate_globals(vector<rover>& fidos, landmark* POIs){
    /// Each POI gives the value of the closest observation to ALL rovers.
    for (int p=0; p< num_POI; p++){
        int closest = POIs[p].find_kth_closest_rover(0, fidos);
        double red = POIs[p].calc_red_observation_value(POIs[p].distances.at(closest));
        double blue = POIs[p].calc_blue_observation_value(POIs[p].distances.at(closest));
        //cout << "REDBLUE: " << red << "\t" << blue << endl;
        for(int r=0; r< num_ROVERS; r++){
        fidos.at(r).global_red_chunks.push_back(red);
        fidos.at(r).global_blue_chunks.push_back(blue);
        }
    }
}

void calculate_differences(vector<rover>& fidos, landmark* POIs){
    /// Calculate Difference Rewards
    for (int p=0; p< num_POI; p++){
        int closest = POIs[p].find_kth_closest_rover(0,fidos);
        int second_closest = POIs[p].find_kth_closest_rover_not_i(0,closest,fidos);
        /// Globals
        double gred = POIs[p].calc_red_observation_value(POIs[p].distances.at(closest));
        double gblue = POIs[p].calc_blue_observation_value(POIs[p].distances.at(closest));
        /// Counterfactuals
        double cred = POIs[p].calc_red_observation_value(POIs[p].distances.at(second_closest));
        double cblue = POIs[p].calc_blue_observation_value(POIs[p].distances.at(second_closest));
        /// Push Back Difference
        fidos.at(closest).difference_red_chunks.push_back(gred - cred);
        fidos.at(closest).difference_blue_chunks.push_back(gblue - cblue);
    }
}

void collect(vector<rover>& fidos, landmark* POIs, int ev){
    /// 1) Assign reward units to policies.
    /// 2) ?? ///@DW
    
    /// 1)
    for (int r = 0; r < num_ROVERS; r++)
    {
        int thisone = fidos.at(r).selected.at(ev);
            if(DO_LOCAL){
                double localred = accumulate(fidos.at(r).local_red_chunks.begin(),fidos.at(r).local_red_chunks.end(),0.0);
                double localblue = accumulate(fidos.at(r).local_blue_chunks.begin(), fidos.at(r).local_blue_chunks.end(),0.0);
                double globalred = accumulate(fidos.at(r).global_red_chunks.begin(), fidos.at(r).global_red_chunks.end(), 0.0);
                double globalblue = accumulate(fidos.at(r).global_blue_chunks.begin(), fidos.at(r).global_blue_chunks.end(), 0.0);
                fidos.at(r).population.at(thisone).clear_raw_local();
                fidos.at(r).population.at(thisone).clear_raw_objectives();
                fidos.at(r).population.at(thisone).set_next_raw_local(localred);
                fidos.at(r).population.at(thisone).set_next_raw_local(localblue);
                fidos.at(r).population.at(thisone).set_next_raw_objective(localred);
                fidos.at(r).population.at(thisone).set_next_raw_objective(localblue);
			}
        
            if(DO_LOCAL || DO_GLOBAL || DO_DIFFERENCE){
                double globalred = accumulate(fidos.at(r).global_red_chunks.begin(), fidos.at(r).global_red_chunks.end(), 0.0);
                double globalblue = accumulate(fidos.at(r).global_blue_chunks.begin(), fidos.at(r).global_blue_chunks.end(), 0.0);
                fidos.at(r).population.at(thisone).clear_raw_global();
                fidos.at(r).population.at(thisone).set_next_raw_global(globalred);
                fidos.at(r).population.at(thisone).set_next_raw_global(globalblue);
            }
        
        if(DO_GLOBAL){
            double globalred = accumulate(fidos.at(r).global_red_chunks.begin(), fidos.at(r).global_red_chunks.end(), 0.0);
            double globalblue = accumulate(fidos.at(r).global_blue_chunks.begin(), fidos.at(r).global_blue_chunks.end(), 0.0);
            fidos.at(r).population.at(thisone).clear_raw_objectives();
            fidos.at(r).population.at(thisone).set_next_raw_objective(globalred);
            fidos.at(r).population.at(thisone).set_next_raw_objective(globalblue);
        }
            
            if(DO_DIFFERENCE){
                //cout << "R! " << r << endl;
                //cout << "DDR: " << fidos.at(r).global_red_chunks.size() << endl;
                //cout << "DDB: " << fidos.at(r).global_blue_chunks.size() << endl;
                
                double differencered = accumulate(fidos.at(r).difference_red_chunks.begin(),fidos.at(r).difference_red_chunks.end(),0.0);
                double differenceblue = accumulate(fidos.at(r).difference_blue_chunks.begin(),fidos.at(r).difference_blue_chunks.end(),0.0);
                fidos.at(r).population.at(thisone).clear_raw_objectives();
                fidos.at(r).population.at(thisone).clear_raw_difference();
                fidos.at(r).population.at(thisone).set_next_raw_difference(differencered);
                fidos.at(r).population.at(thisone).set_next_raw_difference(differenceblue);
                fidos.at(r).population.at(thisone).set_next_raw_objective(differencered);
                fidos.at(r).population.at(thisone).set_next_raw_objective(differenceblue);
			}
    }
    
    if(DO_LC){
        for (int r = 0; r < num_ROVERS; r++) {
            
            double val=0, redval = 0, blueval=0;
            int thisone = fidos.at(r).selected.at(ev);
            redval = fidos.at(r).population.at(thisone).get_raw_objective(0);
            blueval = fidos.at(r).population.at(thisone).get_raw_objective(1);
            val = blueval + redval;
            fidos.at(r).population.at(thisone).set_fitness(val);
        }
    }

	if (DO_HV){
		for (int r = 0; r < num_ROVERS; r++) {
			double val = 0, redval = 0, blueval = 0;
			int thisone = fidos.at(r).selected.at(ev);            redval = fidos.at(r).population.at(thisone).get_raw_objective(0);
            blueval = fidos.at(r).population.at(thisone).get_raw_objective(1);
			val = blueval * redval;
			fidos.at(r).population.at(thisone).set_fitness(val);
		}
	}
    
    /// 3)
    store_policy_locations(fidos);
    for (int i = 0; i < num_ROVERS; i++)
    {
        fidos.at(i).store_x.clear();
        fidos.at(i).store_y.clear();
    }
    
}

void make_random_pois(landmark* POIs){
	for (int p = 0; p < num_POI; p++){
		double x = (rand()%XMAX)+1;
		double y = (rand()%YMAX)+1;
		double red_val = LYRAND * 10;
		double blue_val = LYRAND * 10;
		POIs[p].create(x, y, red_val, blue_val);
	}
}

void set_up_all_pois(landmark* POIs){
    if(POI_GENERATE==0){
        cout << "READING POIs FROM FILE!" << endl;
	int n = 1, n1 = 0, n2 = 0, n3 = 0, n4 = 0;
	double num = 0;
	double x, y, red, blue;

	ifstream datafile("poi_values.txt");
	while (datafile >> num){
		if (n % 4 == 1) x = num, n1++;
		else if (n % 4 == 2) y = num, n2++;
		else if (n % 4 == 3) red = num, n3++;
		else if (n % 4 == 0){
			blue = num;
			POIs[n4].create(x, y, red, blue);
			n4++;
		}
		n++;
	}
        datafile.close();}
    else{
        cout << "MAKING RANDOM POIs!" << endl;
        make_random_pois(POIs);
    }
}

void set_up_all_rovers(vector<rover>& fidos){
    /// x,y,h
    for (int r = 0; r < num_ROVERS; r++)
    {
        fidos.at(r).reset();
        fidos.at(r).population.clear();
        neural_network NN;
         for (int p = 0; p < EVOPOP; p++){
        /// Set up neural network
        if (TELEPORTATION == 0){
            NN.clean();
            NN.setup(INPUTS, HIDDEN, OUTPUTS);
            for (int in = 0; in < INPUTS; in++){
                if (in < 4){
                    NN.take_in_min_max(0, 15);
                }
                if (in >= 4){
                    NN.take_in_min_max(0, 15);
                }
            }
            for (int out = 0; out < OUTPUTS; out++){
                NN.take_out_min_max(-XMAX / 10, XMAX / 10);
            }
        }
        if (TELEPORTATION == 1){
            NN.clean();
            NN.setup(0, HIDDEN, OUTPUTS);
            for (int out = 0; out < OUTPUTS; out++){
                NN.take_out_min_max((double)-0.1*XMAX, (double)1.1*XMAX);
            }
        }
        
        /// create population of neural networks.
       
            fidos.at(r).population.push_back(NN);
            fidos.at(r).selected.push_back(p);
        }
    }
    
    deterministic_and_random_place(fidos);
    //rover_place_test(fidos);
}

int main()
{
    cout << "Hello world!" << endl;
	srand(time(NULL));
    cout << "RANDOM SEED: " << time(NULL) << endl;
    
    if(TELEPORTATION){cout << "TELEPORTATION RUN" << endl;}

	FILE * pFILE1 = fopen("rover_locations2.txt", "w");
	FILE * pFILE2 = fopen("poi_locations.txt", "w");
	FILE * pFILE3 = fopen("fitnesses.txt", "w");
	FILE * pFILE4 = fopen("red_blue_statrun.txt", "w");
    FILE * pFILE5;
    if(POI_GENERATE){
        cout << "GENERATING POIs TO SAVE TO FILE" << endl;
	pFILE5 = fopen("poi_values.txt", "w");
    }

	for (int stat_run = 0; stat_run < STAT_RUN; stat_run++)
	{
		/// BGN Create Landmarks
		landmark POIs[num_POI];

		/// x, y, r, b;
		//POIs[0].create(10, 10, 10, 10);
		//POIs[1].create(10, 90, 10, 10);
		//POIs[2].create(90, 10, 10, 00);
		//POIs[3].create(90, 90, 10, 10);

		//make_random_pois(POIs);
		set_up_all_pois(POIs);

		vector<double> poi_x_locations, poi_y_locations;
		vector<double> poi_red_values, poi_blue_values;

		for (int j = 0; j < num_POI; j++)
		{
			poi_x_locations.push_back(POIs[j].x);
			poi_y_locations.push_back(POIs[j].y);
			poi_red_values.push_back(POIs[j].start_red);
			poi_blue_values.push_back(POIs[j].start_blue);
		}

		print_poi_locations(pFILE2, poi_x_locations, poi_y_locations);
		if(POI_GENERATE)
        {
            print_poi_all_values(pFILE5, poi_x_locations, poi_y_locations, poi_red_values, poi_blue_values);
        }
		/// END Create Landmarks

		/// BGN Create Rovers
		vector<rover> fidos(num_ROVERS);
		//vector<rover>* pfidos = &fidos;
        set_up_all_rovers(fidos);
		/// END Create Rovers

		cout << "Done with rover setup" << endl;

		cout << "Preliminaries completed" << endl;
		for (int gen = 0; gen < GENERATIONS; gen++)
		{
            if(gen % (GENERATIONS/100) == 0){
                //cout << "Beginning Generation " << gen << endl;}
                cout << "Stat run " << stat_run << " is " << (double)gen/GENERATIONS*100  << " % complete" << endl;
            }
            
			for (int r = 0; r < num_ROVERS; r++)
			{
				int SWAPS = 0;
				for (int i = 0; i < SWAPS; i++)
				{
					int p1 = rand() % EVOPOP;
					int p2 = rand() % EVOPOP;
					int holder;

					holder = fidos.at(r).selected.at(p1);
					fidos.at(r).selected.at(p1) = fidos.at(r).selected.at(p2);
					fidos.at(r).selected.at(p2) = holder;
				}
			}

			for (int ev = 0; ev < EVOPOP; ev++)
			{
				clear_rewards(fidos);

				for (int k = 0; k < num_ROVERS; k++)
				{
					fidos.at(k).replace();
				}

				for (int t = 0; t < TIMESTEPS; t++)
				{
					//if(t%100==0){
					//cout << "." << flush;}

					/// SENSE
					//cout << "Sense!" << endl;
					for (int r = 0; r < num_ROVERS; r++)
					{
						fidos.at(r).heading = 0;
						fidos.at(r).sense(POIs, fidos);
					}

					/// DECIDE
					// Run the neural network here.
					//cout << "Decide!" << endl;
					for (int r = 0; r < num_ROVERS; r++)
					{
						fidos.at(r).decide(ev);
					}

					/// ACT
					//cout << "ACT!" << endl;
					if (gen == GENERATIONS - 1 && STAT_RUN == 1 && t==0){
						print_rover_locations(pFILE1, fidos);
						store_timestep_locations(fidos);
					}

					for (int r = 0; r < num_ROVERS; r++)
					{
						fidos.at(r).act();
					}

					if (gen == GENERATIONS - 1 && STAT_RUN == 1){
						print_rover_locations(pFILE1, fidos);
						store_timestep_locations(fidos);
					}

					/// REACT
                    //cout << "REACT!" << endl;
                    complete_react(fidos, POIs);
				} /// END TIMESTEP LOOP
                //cout << "COLLECT!" << endl;
                collect(fidos, POIs, ev); // End of episode cleanup.
			} /// END EVOPOP LOOP
			
			print_red_blue_statrun(pFILE4, fidos, stat_run);
            
            if (DO_CENTRALIZED_NSGA){
                NSGA_2 C_NSGA;
                C_NSGA.declare_NSGA_dimension(2);
                C_NSGA.NSGA_reset();
                int dex=0;
                for (int ev = 0; ev < EVOPOP; ev++) {
                    for(int r=0; r<num_ROVERS; r++){
                    vector<double> afit = fidos.at(r).population.at(ev).get_raw_objectives();
                    C_NSGA.vector_input(afit, dex);
                    dex++;
                    }
                }
                C_NSGA.execute();
                dex=0;
                for (int ev = 0; ev < EVOPOP; ev++) {
                    for(int r=0; r<num_ROVERS; r++){
                    fidos.at(r).population.at(ev).set_fitness(-C_NSGA.NSGA_member_fitness(dex));
                    dex++;
                    }
                }
            }
            
            if(DO_DISTRIBUTED_NSGA){
                vector<NSGA_2> dis_NSGA;
                /// For each rover, create an NSGA calculator.
                for(int r=0; r<num_ROVERS; r++){
                    NSGA_2 N;
                    N.declare_NSGA_dimension(2);
                    N.NSGA_reset();
                    dis_NSGA.push_back(N);
                }
                /// For each rover, input all fitnesses into the right NSGA calculator.
                for(int r=0; r<num_ROVERS; r++){
                    for (int ev = 0; ev < EVOPOP; ev++) {
                    vector<double> afit = fidos.at(r).population.at(ev).get_raw_objectives();
                    dis_NSGA.at(r).vector_input(afit,ev);
                    }
                    dis_NSGA.at(r).execute();
                    for (int ev=0; ev < EVOPOP; ev++) {
                        fidos.at(r).population.at(ev).set_fitness(-dis_NSGA.at(r).NSGA_member_fitness(ev));
                    }
                }
            }
            
            if(DO_D_OF_NSGA_DISTRIBUTED){
                for(int r=0; r<num_ROVERS; r++){
                    /// Preliminaries.
                    vector<double> Gs;
                    vector<double> Gzmis;
                    vector<double> Ds;
                    /// Calculate G for all population members.
                    NSGA_2 G;
                    G.declare_NSGA_dimension(2);
                    G.NSGA_reset();
                    for (int ev = 0; ev < EVOPOP; ev++) {
                        vector<double> afit = fidos.at(r).population.at(ev).get_raw_objectives();
                        vector<double> gzmifit = fidos.at(r).population.at(ev).get_raw_gzmis();
                        G.vector_input(afit,ev);
                    }
                    G.execute();
                    for(int ev=0; ev< EVOPOP; ev++){
                        Gs.push_back(-G.NSGA_member_fitness(ev));
                    }
                    /// Calculate gzmi for all population members.
                    for(int ev=0; ev<EVOPOP; ev++){
                        NSGA_2 gzmi;
                        gzmi.declare_NSGA_dimension(2);
                        gzmi.NSGA_reset();
                        for(int ev2=0; ev2<EVOPOP; ev2++){
                            vector<double> afit = fidos.at(r).population.at(ev).get_raw_objectives();
                            vector<double> gzmifit = fidos.at(r).population.at(ev).get_raw_gzmis();
                            if(ev2==ev){
                                gzmi.vector_input(gzmifit,ev);
                            }
                            else{
                                gzmi.vector_input(afit,ev);
                            }
                        }
                        gzmi.execute();
                        for(int ev2=0; ev2<EVOPOP; ev2++){
                            Gzmis.push_back(-gzmi.NSGA_member_fitness(ev2));
                        }
                    }
                    /// Calculate D for all population members
                    for(int ev=0; ev<EVOPOP; ev++){
                        Ds.push_back(Gs.at(ev)-Gzmis.at(ev));
                    }
                }
            } /// END DO_D_OF_NSGA_DISTRIBUTED
            
			/*
			SPEA_2 SPEA;

			if (DO_SPEA){
			SPEA.vector_input(MO, a);
			}

			if (DO_SPEA){
			for (int a = 0; a<pVA->size(); a++){
			vector<double> MO;
			MO.push_back(pVA->at(a).get_f1());
			MO.push_back(pVA->at(a).get_f2());
			SPEA.vector_input(MO, a);
			SPEA.take_agent(pVA->at(a), a);
			}

			vector<int> survivors;
			vector<int>* pS = &survivors;
			SPEA.execute(pS);

			pVA->clear();
			for (int i = 0; i< pS->size(); i++){
			int el = pS->at(i);
			pVA->push_back(SPEA.archive.at(el).agent);
			pVA->back().mutate();
			}
			}
			*/

            if(FITNESS_FILE_WATCH){
			print_fitnesses(pFILE3, fidos, gen);
            }
            
			for (int r = 0; r < num_ROVERS; r++)
			{
				vector<neural_network>* pVNN = &fidos.at(r).population;
				vector<double> fit = kill_lowest_performers(pVNN, r, fidos);
				expand_population(pVNN, r, fit);
			}

			for (int r = 0; r < num_ROVERS; r++)
			{
				for (int i = 0; i < EVOPOP; i++)
				{
					fidos.at(r).population.at(i).clean();
				}
			}

		} /// END GENERATION LOOP.
	}
	fclose(pFILE1);
	fclose(pFILE2);
	fclose(pFILE3);
	fclose(pFILE4);
    if(POI_GENERATE){
	fclose(pFILE5);
    }
	return 0;
}



vector<double> kill_lowest_performers(vector<neural_network>* pNN, int r, vector<rover>& fidos){
	//vector<int> kill;
	/// r is the rover population we are working with.
	/// We kill the 'n' lowest performing NNs (mark for replacement in expand_population)

	// first, assemble a vector of these neural network fitnesses.
	vector<double> fitnesses;
	for (int i = 0; i<EVOPOP; i++){
		fitnesses.push_back(pNN->at(i).get_fitness());
	}

	/// for as many as we need to eliminate...
	for (int rep = 0; rep<(EVOPOP / 2); rep++){
		/// find the lowest index, 1 at a time.
		double lowest_fitness = 99999999999999; /// high fitness;
		int lowest_dex = -1; /// dummy index;
		for (int i = 0; i<fitnesses.size(); i++){
			double debug_1 = fitnesses.at(i);
			double debug_2 = fitnesses.size();
			if (fitnesses.at(i) < lowest_fitness){
				lowest_fitness = fitnesses.at(i);
				lowest_dex = i;
			}
		}
		/// kill lowest fitness.
		fitnesses.erase(fitnesses.begin() + lowest_dex);
		/// kill matching population member.
		pNN->erase(pNN->begin() + lowest_dex);
		//kill.push_back(lowest_fitness);
	}

	//    return kill;
	return fitnesses;
}

void expand_population(vector<neural_network>* pNN, int r, vector<double> fitnesses){
	//vector<int> survivors;
	/// create new neural networks like the survivors.
	for (int i = 0; i<EVOPOP / 2; i++){
		int spot; // the index of the one we're replicating.
		if (LYRAND < 0.8) {/// THIS ONE SELECTS THE BEST TO REPLICATE
			spot = max_element(fitnesses.begin(), fitnesses.end()) - fitnesses.begin();
		}
		else {
			/// THIS ONE SELECTS A RANDOM SURVIVOR TO REPLICATE.
			spot = rand() % pNN->size();
		}
		pNN->push_back(pNN->at(spot));
		pNN->back().mutate();
	}
}