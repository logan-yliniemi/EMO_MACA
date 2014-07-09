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
#include "NNV.h"

#define pi 3.141529
#define QUADRANTS 4
#define XMIN 0
#define XMAX 100
#define YMIN 0
#define YMAX 100

#define num_POI 4
#define num_ROVERS 3
#define DETERMINISTICALLY_PLACED 3 //cannot be more than num_ROVERS

#define TIMESTEPS 10
#define GENERATIONS 1

// NEURAL NETWORK PARAMETERS
#define INPUTS 8
#define HIDDEN 10
#define OUTPUTS 2
#define EVOPOP 100

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
	double xdot;
	double ydot;
	int ID;
	double rover_state[QUADRANTS];
	double blue_state[QUADRANTS];
	double red_state[QUADRANTS];

	int basic_sensor(double, double, double, double, double);
	void reset();
	int place(double, double, double);
	double strength_sensor(double, double, double);
	void move();
	void full_red_sensor(landmark*);
	void full_blue_sensor(landmark*);
	void full_rover_sensor(vector<rover>);


	double local_red;
	double local_blue;
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

	void create(double, double, double, double);
	void reset();


	int find_kth_closest_rover(int, vector<rover>);
	double find_dist_to_rover(int, vector<rover>);
	int find_kth_closest_rover_not_i(int, int, vector<rover>);

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
	min_obs_distance = XMAX / 100; /// LYLY ADJUSTABLE
	max_obs_distance = XMAX / 1; /// LYLY ADJUSTABLE
}

void landmark::reset()
{
	red_value = start_red;
	blue_value = start_blue;
}

int landmark::find_kth_closest_rover(int k, vector<rover> fidos)
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

double landmark::find_dist_to_rover(int rvr, vector<rover> fidos)
{
	double delx, dely;
	delx = fidos.at(rvr).x - x;
	dely = fidos.at(rvr).y - y;
	double dis = sqrt(delx*delx + dely*dely);

	return dis;
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

int deterministic_place(vector<rover>& fidos)
{
	srand(1);
	double x, y, heading;
	for (int i = 0; i < DETERMINISTICALLY_PLACED; i++)
	{
		x = rand() % 101;
		y = rand() % 101;
		heading = rand() % 361 * pi / 180;
		cout << x << " " << y << " " << heading << endl;
		fidos.at(i).place(x, y, heading);
	}

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
	double str = -1;

	double delta = find_distance(x, y, tarx, tary);

	double tarheading = atan2((y - tary), (x - tarx));

	double del_heading;
	del_heading = tarheading - heading;
	angle_resolve_pmpi(del_heading);

	//cout << "del_heading: " << del_heading << endl;

	double nw = pi / 4;
	double ne = -pi / 4;
	double sw = 3 * pi / 4;
	double se = -3 * pi / 4;
	double center;

	if (del_heading<nw && del_heading>ne) /// +/- 45
	{
		center = 0;
	}
	if (del_heading >= nw && del_heading<sw)
	{
		/// object is "left" of the robot
		center = pi / 2;
	}
	if (del_heading <= ne && del_heading>se)
	{
		///object is "right" of the robot
		center = -pi / 2;
	}
	if (del_heading <= se || del_heading >= sw)
	{
		///object is "behind" the robot
		center = pi;
	}

	double theta = center - del_heading;

	if (center == pi)
	{
		theta = fmin(theta, -center - del_heading);
	}

	str = value / delta*(1 - theta / (pi / 4));

	return str;
}

void rover::full_red_sensor(landmark* POIs)
{
	int quadrant;
	for (int i = 0; i<num_POI; i++)
	{
		quadrant = basic_sensor(x, y, heading, POIs[i].x, POIs[i].y);
		double value = POIs[i].red_value;
		double tarx = POIs[i].x;
		double tary = POIs[i].y;
		double str = strength_sensor(value, tarx, tary);

		red_state[quadrant] += str;
	}
}

void rover::full_blue_sensor(landmark* POIs)
{
	int quadrant;
	for (int i = 0; i<num_POI; i++)
	{
		quadrant = basic_sensor(x, y, heading, POIs[i].x, POIs[i].y);
		double value = POIs[i].blue_value;
		double tarx = POIs[i].x;
		double tary = POIs[i].y;
		double str = strength_sensor(value, tarx, tary);

		blue_state[quadrant] += str;
	}
}

void rover::full_rover_sensor(vector<rover> fidos)
{
	int quadrant;
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

FILE* open_files(){
    FILE* pFILE = fopen ("Testing.txt","w");
    return pFILE;
}

vector<double> kill_lowest_performers(vector<neural_network>* pNN, int r);
void expand_population(vector<neural_network>* pNN, int r, vector<double>);

void print_rover_locations(FILE * dpFILE, double x1, double y1, double x2, double y2, double x3, double y3)
{
	fprintf(dpFILE, "%.2f %.2f %.2f %.2f %.2f %.2f\n", x1, y1, x2, y2, x3, y3);
}

void print_poi_locations(FILE * ddpFILE, vector<double> x, vector<double> y, double POIs)
{
	for (int i = 0; i < POIs; i++)
	{
		fprintf(ddpFILE, "%.2f %.2f\n", x.at(i), y.at(i));
	}
}

int main()
{
	cout << "Hello world!" << endl;
	srand(time(NULL));
//        basic_rover_sensor_testing();
//        FILE* pFILE = open_files();
	
	FILE * dpFILE = fopen("rover_locations.txt", "w");
	FILE * ddpFILE = fopen("poi_locations.txt", "w");
		
	///* commented out for testing
	landmark POIs[num_POI];
	//vector<landmark> POIs(num_POI);

	/// x, y, r, b;
	POIs[0].create(10, 10, 10, 10);
	POIs[1].create(10, 90, 0, 100);
	POIs[2].create(90, 10, 100, 0);
	POIs[3].create(90, 90, 100, 100);
	/*
	for (int i = 0; i < num_POI; i++)
	{
		POIs.at(i).create(
	}
	*/

	vector<double> poi_x_locations;
	vector<double> poi_y_locations;

	for (int j = 0; j < num_POI; j++)
	{
		poi_x_locations.push_back(POIs[j].x);
		poi_y_locations.push_back(POIs[j].y);
	}

	print_poi_locations(ddpFILE, poi_x_locations, poi_y_locations, num_POI);

        vector< vector<neural_network> > VVNN;
        vector< neural_network > VNN; 
        neural_network NN;
        
        VVNN.clear();
        for (int r = 0; r<num_ROVERS; r++)
	{
            VNN.clear();
        for (int i = 0; i<EVOPOP; i++)
	{
            NN.clean();
            NN.setup(INPUTS,HIDDEN,OUTPUTS);
            for(int in=0; in<INPUTS; in++){
                if(in<4){
                    NN.take_in_min_max(0,num_ROVERS);
                }
                if(in>=4){
                    NN.take_in_min_max(0,500);
                }
            }
            for (int out = 0; out<OUTPUTS; out++){
                NN.take_out_min_max(-XMAX / 10,XMAX / 10);
            }
            VNN.push_back(NN);
        }
        VVNN.push_back(VNN);
        }
        
//	//neural_network NN[num_ROVERS][EVOPOP];
//	//vector<double> mini, maxi, mino, maxo;
//	for (int in = 0; in<INPUTS; in++)
//	{
//            for (int r = 0; r<num_ROVERS; r++)
//	{
//                for (int i = 0; i<EVOPOP; i++)
//		{
//                    if(in<4){
//                    VVNN.at(r).at(i).setup(INPUTS,HIDDEN,OUTPUTS);
//                    VVNN.at(r).at(i).take_in_min_max(0,num_ROVERS);
//                    }
//                    if(in>=4){
//                    VVNN.at(r).at(i).setup(INPUTS,HIDDEN,OUTPUTS);
//                    VVNN.at(r).at(i).take_in_min_max(0,500);
//                    }
//                }
//        }
//		//mini.push_back(0);
//		//if (i<4)
//		//{
//                    
//		//	maxi.push_back(num_ROVERS);    /// TODO Generalize
//		//}
//	//	if (i >= 4)
//	//	{
//	//		maxi.push_back(500);    /// TODO Generalize
////		}
//	}
//	for (int out = 0; out<OUTPUTS; out++)
//	{
//            for (int r = 0; r<num_ROVERS; r++)
//	{
//		for (int i = 0; i<EVOPOP; i++)
//		{
//            //NN[r][i].take_out_min_max(-XMAX / 10,XMAX / 10);
//		//mino.push_back(-XMAX / 10);
//		//maxo.push_back(XMAX / 10);
//                }
//            }
//	}
	cout << "done with inputsoutputs scaling" << endl;

	//for (int r = 0; r<num_ROVERS; r++)
	//{
//		for (int i = 0; i<EVOPOP; i++)
//		{
                        //NN[r][i].setup(INPUTS,HIDDEN,OUTPUTS);
                        //NN[r][i].take_in_min_max(mini,maxi);
                        //NN[r][i].take_out_min_max(mino,maxo);
//		}
//	}
	cout << "nn accepted scaling factors" << endl;
	vector<rover> fidos(num_ROVERS);
	/// x,y,h
	for (int r = 0; r<num_ROVERS; r++)
	{
		fidos.at(r).reset();
	}

	deterministic_place(fidos);
	//random_place();

	int selected[num_ROVERS][EVOPOP];

	cout << "Preliminaries completed" << endl;
	for (int gen = 0; gen<GENERATIONS; gen++)
	{
		cout << "Beginning Generation " << gen << endl;
		for (int ev = 0; ev<EVOPOP; ev++)
		{
			for (int k = 0; k < num_ROVERS; k++)
			{
				fidos.at(k).local_blue = 0;
				fidos.at(k).local_red = 0;
			}

			for (int t = 0; t<TIMESTEPS; t++)
			{
				//if(t%100==0){
				//cout << "." << flush;}

				for (int r = 0; r<num_ROVERS; r++)
				{
					for (int ev = 0; ev<EVOPOP; ev++)
					{
						selected[r][ev] = ev;
					}
					int SWAPS = 100;
					for (int i = 0; i<SWAPS; i++)
					{
						int storage;
						int p1 = rand() % EVOPOP;
						int p2 = rand() % EVOPOP;
						storage = selected[r][p1];
						selected[r][p1] = selected[r][p2];
						selected[r][p2] = storage;
					}
				}
				/// SENSE
				//cout << "Sense!" << endl;		
				for (int r = 0; r<num_ROVERS; r++)
				{
					//cout << "sensing " << r << endl;
					fidos.at(r).full_blue_sensor(POIs);
					fidos.at(r).full_red_sensor(POIs);
					fidos.at(r).full_rover_sensor(fidos);
					//cout << "end sensing " << r << endl;
				}


				/// DECIDE
				// Run the neural network here.
				//cout << "Decide!" << endl;
				for (int r = 0; r<num_ROVERS; r++)
				{
					vector<double> inp;
					//inp.push_back(fidos[r].x);
					//inp.push_back(fidos[r].y);
					for (int i = 0; i<QUADRANTS; i++)
					{
						inp.push_back(fidos.at(r).red_state[i]);
					}
					for (int i = 0; i<QUADRANTS; i++)
					{
						inp.push_back(fidos.at(r).blue_state[i]);
					}

					//for(int i=0; i<EVOPOP; i++)
					//{
                                        VVNN.at(r).at(selected[r][ev]).clean();
                                        VVNN.at(r).at(selected[r][ev]).take_vector_input(inp);
					//NN[r][selected[r][ev]].readinputs(inp);
					//}

					//for(int i=0; i<EVOPOP; i++)
					//{
					//NN[i].scaleinputs();
					//NN[r][selected[r][ev]].go();
					VVNN.at(r).at(selected[r][ev]).execute(INPUTS,OUTPUTS);
					//NN[i].scaleoutputs();
					//}
					//for(int i=0; i<EVOPOP; i++)
					//{
                                        //cout << "MAXO 0: " << maxo.at(0) << endl;
                                        //cout << "MAXO 1: " << maxo.at(1) << endl;
					fidos.at(r).xdot = VVNN.at(r).at(selected[r][ev]).give_output(0);
                                                //output[0];
					fidos.at(r).ydot = VVNN.at(r).at(selected[r][ev]).give_output(1);
                                                //output[1];
                                        //cout << NN[r][selected[r][ev]].output[0] << endl;
                                        //cout << "FIDODX " << fidos[r].xdot << endl;
                                        //cout << "FIDODY " << fidos[r].ydot << endl;
					//}
				}


				/// ACT
				//cout << "ACT!" << endl;

				print_rover_locations(dpFILE, fidos.at(0).x, fidos.at(0).y, fidos.at(1).x, fidos.at(1).y, fidos.at(2).x, fidos.at(2).y);

				for (int r = 0; r<num_ROVERS; r++)
				{
					//cout << "acting " << r << endl;
                                    //cout << "fidos x " << fidos[r].x << "\t";
					fidos.at(r).move();
                                    //cout << "fidos y " << fidos[r].y << endl;
					//cout << "end acting " << r << endl;
				}

				/// REACT
				//cout << "REACT!" << endl;
				for (int i = 0; i<num_POI; i++)
				{
					//cout << "begin react " << i << endl;
					int assignee = POIs[i].find_kth_closest_rover(0, fidos);
					double distance = POIs[i].find_dist_to_rover(assignee, fidos);
					double red_observation_value = POIs[i].calc_red_observation_value(distance);
					double blue_observation_value = POIs[i].calc_blue_observation_value(distance);

					fidos.at(assignee).local_red += red_observation_value;
					fidos.at(assignee).local_blue += blue_observation_value;

					//cout << "at distance " << distance << endl;
					//cout << "red value of " << red_observation_value << " assigned to rover " << assignee << endl;
					//cout << "blue value of " << blue_observation_value << " assigned to rover " << assignee << endl;
					//cout << "end react " << i << endl;

					//int assignee_not_i=POIs[i].find_kth_closest_rover_not_i(0,i,&fido);
					//double distance_not_i=POIs[i].find_dist_to_rover(assignee_not_i,&fido);
				}
				//cout << "end timestep" << endl;
			}
			/// END TIMESTEP

			for (int r = 0; r<num_ROVERS; r++)
			{
				for (int ev = 0; ev<EVOPOP; ev++)
				{
					//VVNN.at(r).at(selected[r][ev]).set_fitness(fidos[r].local_red + fidos[r].local_blue);
					VVNN.at(r).at(selected[r][ev]).set_fitness(fidos.at(r).local_red + fidos.at(r).local_blue);
					//cout << fidos[r].local_red << " " << fidos[r].local_blue << endl;
				}
			}

		}
		/// END EVOPOP LOOP

		cout << "This generation's best local fitness is: " << VVNN.at(0).at(selected[0][0]).get_fitness() << endl;

		for (int r = 0; r<num_ROVERS; r++)
		{
                    vector<neural_network>* pVNN = &VVNN.at(r);
                    vector<double> fit = kill_lowest_performers(pVNN,r);
                    //sort_by_rank(NN,r);
                    //evolve_ranked_population(NN,r);
                    expand_population(pVNN,r,fit);
			//NN[r][0].ranker(NN[r]);
			//NN[r][0].sorter(NN[r]);
		}

		for (int r = 0; r<num_ROVERS; r++)
		{
			for (int i = 0; i<EVOPOP; i++)
			{
//				NN[r][i].evolve(NN[r], i);
			}
		}
                for (int r = 0; r<num_ROVERS; r++)
		{
			for (int i = 0; i<EVOPOP; i++)
			{
                            VVNN.at(r).at(i).clean();
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