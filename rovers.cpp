// Multi-Objective Multi-Agent Rover Domain.
// For use with Difference Rewards in SPEA2 and NSGA-II
// Core code: Logan Yliniemi, 2012.
// NSGAheader, SPEAheader, NNLIBv2, Logan Yliniemi, 2013.
// Alterations: Logan Yliniemi and Drew T. Wilson, July 2014

#include <iostream>
#include <math.h>
#include <vector>
#include "NNLIBv2.h"
#include <ctime>

#define pi 3.141529
#define QUADRANTS 4
#define XMIN 0
#define XMAX 100
#define YMIN 0
#define YMAX 100

#define num_POI 4
#define num_ROVERS 3

#define TIMESTEPS 100
#define GENERATIONS 1000

///NEURAL NETWORK PARAMETERS (IN NN HEADER)
//#define INPUTS 10
//#define HIDDEN 5
//#define OUTPUTS 1
//#define EVOPOP 10

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
	void full_rover_sensor(rover*);


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


	int find_kth_closest_rover(int, rover*);
	double find_dist_to_rover(int, rover*);
	int find_kth_closest_rover_not_i(int, int, rover*);

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
	max_obs_distance = XMAX / 10; /// LYLY ADJUSTABLE
}

void landmark::reset()
{
	red_value = start_red;
	blue_value = start_blue;
}

int landmark::find_kth_closest_rover(int k, rover* fidos)
{
    int kclosest;
    double closest_distance;
    vector<double> distances;
    for(int b=0; b<num_ROVERS; b++){
        double delx, dely;
        delx = fidos[b].x - x;
	dely = fidos[b].y - y;
        double dis = sqrt(delx*delx + dely*dely);
        distances.push_back(dis);
    }
    vector<double> distances_unsorted = distances;
    
    sort(distances.begin(),distances.end());
    closest_distance=distances.at(k);
    for(int b=0; b<num_ROVERS; b++){
        if(distances_unsorted.at(b)==closest_distance){
            kclosest=b;
            break;
        }
    }
    return kclosest;
}

int landmark::find_kth_closest_rover_not_i(int k, int i, rover* fidos){
    int kclosest;
    double closest_distance;
    vector<double> distances;
    for(int b=0; b<num_ROVERS; b++){
        if(b==i){distances.push_back(XMAX*2+YMAX*2); continue;} /// Infeasible long distance.
        double delx, dely;
        delx = fidos[b].x - x;
	dely = fidos[b].y - y;
        double dis = sqrt(delx*delx + dely*dely);
        distances.push_back(dis);
    }
    vector<double> distances_unsorted = distances;
    
    sort(distances.begin(),distances.end());
    closest_distance=distances.at(k);
    for(int b=0; b<num_ROVERS; b++){
        if(distances_unsorted.at(b)==closest_distance){
            kclosest=b;
            break;
        }
    }
    return kclosest;
}

double landmark::find_dist_to_rover(int rvr, rover* fidos)
{
	double delx, dely;
	delx = fidos[rvr].x - x;
	dely = fidos[rvr].y - y;
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
	heading = atan2(xdot, ydot);
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

int rover::basic_sensor(double roverx, double rovery, double rover_heading, double tarx, double tary)
{
        /// @DW: Ensure that sensor rotates with rover.
        /// @DW: Not touching any of the comments 7/7 to not break it before testing. Should be cleaned up right after.
    
	/// Create a square and determine whether or not the rover falls into this square.
	//double roverx;
	//double rovery;
	//double rover_heading;

	///TEMPORARY
	//roverx=10;
	//rovery=10;
	//rover_heading=0;
	///TEMPORARY

	//double rover_right=rover_heading+pi/2;
	//double rover_left=rover_heading+3*pi/2;
	//double rover_reverse=rover_heading+pi;
	//angle_resolve(rover_right);
	//angle_resolve(rover_left);
	//angle_resolve(rover_reverse);

	//double tarx;
	//double tary;

	///TEMPORARY
	//tarx=20;
	//tary=15;
	///TEMPORARY

	/// start from rover x,y. If thing is to the right of rover_heading, then do this case.
	// (roverx,rovery,roverheading) describes the line.
	// (tarx,tary) describes our target.
	// for the thing to be to the right of the rover ray, if we have a positive angle, we should first find the distance between the two.

	//double dist;
	double dx;
	double dy;
	//dx=roverx-tarx;
	//dy=rovery-tary;

	dx = tarx - roverx;
	dy = tary - rovery;

	//dist=dx*dx+dy*dy;
	//dist=sqrt(dist);

	// heading to target with respect to robot frame
	double tarheading;
	tarheading = atan2(dx, dy);

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

	//if(del_heading>0 && del_heading<pi/2)
	//{
	//    return 3;
	//    /// in sector 0;
	//}
	//if(del_heading>pi/2 && del_heading<pi)
	//{
	//    return 1;
	//    /// in sector 1;
	//}
	//if(del_heading>pi && del_heading<3*pi/2)
	//{
	//    return 2;
	//    /// in sector 2;
	//}
	//if(del_heading>3*pi/2 && del_heading<2*pi)
	//{
	//    return 3;
	//    /// in sector 3;
	//}

	else
	{
		cout << "problems in basic_sensor;" << endl;
		return 66;
	}

	/// if thing is to the front of rover_right, then it is case 0.
	/// if thing is to the back of rover_right, then it is case 1.
        
	/// if thing is to the left of rover_heading, then do this case.
	/// if thing is to the front of rover_left, then it is case 3.
	/// if thing is to the back of rover_left, then it is case 2.

	/// If not applied to any quadrant, then return an error.
}

double rover::strength_sensor(double value, double tarx, double tary)
{
	double str = -1;

	double delta = find_distance(x, y, tarx, tary);

	double tarheading = atan2((x - tarx), (y - tary));

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

void rover::full_rover_sensor(rover* fidos)
{
	int quadrant;
	for (int r = 0; r<num_ROVERS; r++)
	{
		if (r == ID)
		{
			continue;
		}
		quadrant = basic_sensor(x, y, heading, fidos[r].x, fidos[r].y);
		double value = 1;
		double tarx = fidos[r].x;
		double tary = fidos[r].y;
		double str = strength_sensor(value, tarx, tary);

		rover_state[quadrant] += str;
	}
}

int main()
{
	cout << "Hello world!" << endl;
	srand(time(NULL));

	landmark POIs[num_POI];

	/// x, y, r, b;
	POIs[0].create(10, 10, 10, 10);
	POIs[1].create(10, 90, 0, 100);
	POIs[2].create(90, 10, 100, 0);
	POIs[3].create(90, 90, 100, 100);

	neural NN[num_ROVERS][EVOPOP];
	vector<double> mini, maxi, mino, maxo;
	for (int i = 0; i<INPUTS; i++)
	{
		mini.push_back(0);
		if (i<4)
		{
			maxi.push_back(num_ROVERS);    /// TODO Generalize
		}
		if (i >= 4)
		{
			maxi.push_back(500);    /// TODO Generalize
		}
	}
	for (int i = 0; i<OUTPUTS; i++)
	{
		mino.push_back(0);
		maxo.push_back(XMAX / 10);
	}
	cout << "done with inputsoutputs scaling" << endl;

	for (int r = 0; r<num_ROVERS; r++)
	{
		for (int i = 0; i<EVOPOP; i++)
		{
			NN[r][i].take_limits(mini, maxi, mino, maxo);
		}
	}
	cout << "nn accepted scaling factors" << endl;

	rover fidos[num_ROVERS];
	/// x,y,h
	for (int r = 0; r<num_ROVERS; r++)
	{
		fidos[r].reset();
	}
	fidos[0].place(85, 90, 0);
	fidos[1].place(25, 20, 0);
	fidos[2].place(15, 10, pi / 2);

	int selected[num_ROVERS][EVOPOP];

	cout << "Preliminaries completed" << endl;
	for (int gen = 0; gen<GENERATIONS; gen++)
	{
            /// TODO: Coevolution
		cout << "Beginning Generation " << gen << endl;
		for (int ev = 0; ev<EVOPOP; ev++)
		{
			fidos[0].local_blue = 0;
			fidos[0].local_red = 0;
			fidos[1].local_blue = 0;
			fidos[1].local_red = 0;
			fidos[2].local_blue = 0;
			fidos[2].local_red = 0;
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

				/*for(int i=0; i<num_POI; i++)
				{
				cout << "POI #" << i << endl;
				for(int r=0; r<num_ROVERS; r++)
				{
				fidos[r].basic_sensor(fidos[r].x,fidos[r].y,fidos[r].heading,POIs[i].x,POIs[i].y);
				fidos[r].strength_sensor(POIs[i].red_value, POIs[i].x, POIs[i].y);
				fidos[r].strength_sensor(POIs[i].blue_value, POIs[i].x, POIs[i].y)
				//cout << fidos[r].basic_sensor(fidos[r].x,fidos[r].y,fidos[r].heading,POIs[i].x,POIs[i].y) << endl;
				//cout << "RED: " << fidos[r].strength_sensor(POIs[i].red_value, POIs[i].x, POIs[i].y) << endl;
				//cout << "BLUE: " << fidos[r].strength_sensor(POIs[i].blue_value, POIs[i].x, POIs[i].y) << endl;
				}

				}*/

				/// SENSE
				//cout << "Sense!" << endl;
				// for all rovers
				for (int r = 0; r<num_ROVERS; r++)
				{
					//cout << "sensing " << r << endl;
					fidos[r].full_blue_sensor(POIs);
					fidos[r].full_red_sensor(POIs);
					fidos[r].full_rover_sensor(fidos);
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
						inp.push_back(fidos[r].red_state[i]);
					}
					for (int i = 0; i<QUADRANTS; i++)
					{
						inp.push_back(fidos[r].blue_state[i]);
					}

					//for(int i=0; i<EVOPOP; i++)
					//{
					NN[r][selected[r][ev]].readinputs(inp);
					//}

					//for(int i=0; i<EVOPOP; i++)
					//{
					//NN[i].scaleinputs();
					NN[r][selected[r][ev]].go();
					//NN[i].scaleoutputs();
					//}
					//for(int i=0; i<EVOPOP; i++)
					//{
					fidos[r].xdot = NN[r][selected[r][ev]].output[0];
					fidos[r].ydot = NN[r][selected[r][ev]].output[1];
					//}
				}


				/// ACT
				//cout << "ACT!" << endl;
				for (int r = 0; r<num_ROVERS; r++)
				{
					//cout << "acting " << r << endl;
					fidos[r].move();
					//cout << "end acting " << r << endl;
				}

				/// REACT
				//cout << "REACT!" << endl;
				for (int i = 0; i<num_POI; i++)
				{
					//cout << "begin react " << i << endl;
					int assignee = POIs[i].find_kth_closest_rover(1, fidos);
					double distance = POIs[i].find_dist_to_rover(assignee, fidos);
					double red_observation_value = POIs[i].calc_red_observation_value(distance);
					double blue_observation_value = POIs[i].calc_blue_observation_value(distance);

					fidos[assignee].local_red += red_observation_value;
					fidos[assignee].local_blue += blue_observation_value;

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
					NN[r][selected[r][ev]].fitness += fidos[r].local_red + fidos[r].local_blue;
				}
			}
                        
                        /// END EPISODE
		}

		cout << "This generation's best local fitness is: " << NN[0][selected[0][0]].fitness << endl;

		for (int r = 0; r<num_ROVERS; r++)
		{
			NN[r][0].ranker(NN[r]);
			NN[r][0].sorter(NN[r]);
		}

		for (int r = 0; r<num_ROVERS; r++)
		{
			for (int i = 0; i<EVOPOP; i++)
			{
				NN[r][i].evolve(NN[r], i);
			}
		}

                /// END GENERATION
	}

	//*/
	return 0;
}
