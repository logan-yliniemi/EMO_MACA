/*
* NNLIB.h
*
*  Created on: Jun 9, 2011
*      Author: ylinieml
*/

#ifndef NNLIB_H_
#define NNLIB_H_


//#include <iostream>
//#include <math.h>
//#include <time.h>
//#include <stdio.h>
//#include <stdlib.h> /// PC SIDE ONLY

using namespace std;

#define INPUTS 8
#define HIDDEN 10
#define OUTPUTS 2
#define EVOPOP 100

#define SMALLISH .5

#define EPOCHS 5000
#define STEPS 5

#define LYRAND (double)rand()/RAND_MAX
#define STARTWEIGHTSTEP .05 //.2??
#define SURVIVORPOP 50

#define COBALT 101

#define DEBUG 0
#define ALTERED 50

class neural
{
protected:
	double scaledinput[INPUTS];
	double hiddenin[INPUTS];
	double hiddenout[HIDDEN];
	double outputin[HIDDEN];
	double outputout[OUTPUTS];

	double weightIH[INPUTS][HIDDEN];
	double weightHO[HIDDEN][OUTPUTS];
	double biasweightIH[HIDDEN];
	double biasweightHO[OUTPUTS];

	double sigmoid(double);
	double inputtohidden(int); //hidden sigmoid
	double hiddentoout(int); //output sigmoid

	void scaleinputs();
	void scaleoutputs();

	double weightstep;
	void changeweightstep();

public:

	double input[INPUTS];
	double output[OUTPUTS];

	double fitness;
	int rank;
	double error;

	double getoutputs();

	/// LIMITS
	double maxoutput[OUTPUTS];
	double minoutput[OUTPUTS];

	double maxinput[INPUTS];
	double mininput[INPUTS];
	/// LIMITS

	double initialize();
	double start();
	double reset();

	void score();
	void evolve(neural*, int);

	void readinputs(vector<double>);//, double, double);
	void take_limits(vector<double>, vector<double>, vector<double>, vector<double>);

	double go();

	void ranker(neural*);
	void sorter(neural*);
	void swap(neural*, int, int);

	void printresult();

	void copier(neural*);

	int taken;
};

void neural::copier(neural* a)
{
	error = a->error;
	fitness = a->fitness;
	rank = a->rank;
	weightstep = a->weightstep;
	taken = a->taken;

	for (int hid = 0; hid<HIDDEN; hid++)
	{
		biasweightIH[hid] = a->biasweightIH[hid];
		hiddenout[hid] = a->hiddenout[hid];
		outputin[hid] = a->outputin[hid];
		for (int in = 0; in<INPUTS; in++)
		{
			weightIH[in][hid] = a->weightIH[in][hid];
		}
		for (int out = 0; out<OUTPUTS; out++)
		{
			weightHO[hid][out] = a->weightHO[hid][out];
		}
	}

	for (int out = 0; out<OUTPUTS; out++)
	{
		maxoutput[out] = a->maxoutput[out];
		minoutput[out] = a->minoutput[out];
		biasweightHO[out] = a->biasweightHO[out];
		output[out] = a->output[out];
		outputout[out] = a->outputout[out];
	}
	for (int in = 0; in<INPUTS; in++)
	{
		input[in] = a->input[in];
		maxinput[in] = a->maxinput[in];
		mininput[in] = a->mininput[in];
		scaledinput[in] = a->scaledinput[in];
		hiddenin[in] = a->hiddenin[in];
	}
}

void neural::changeweightstep()
{
	static int calls;
	if (calls % 200>100)
	{
		weightstep *= 99;
	}
	else if (calls % 200<100)
	{
		weightstep /= .99;
	}
}

void neural::printresult()
{
	cout << scaledinput[0] << "->" << outputout[0] << "\t::\t" << input[0] << "->" << output[0] << "\t ::: " << fitness << endl;
}

void neural::readinputs(vector<double> a)
{
	static int counter;
	//for(int i=0; i<INPUTS; i++)
	//{
	//  input[i]=num;
	//}
	for (int i = 0; i<a.size(); i++)
	{
		input[i] = a.at(i);
	}

	//input[0]=zero;
	//input[1]=one;
	//input[2]=two;
	//input[3]=three;
	//input[4]=four;
	//input[5]=five;
	counter++;
	//if(counter%100==0){cout << "counter: " << counter << endl;}
}

#define SCALINGFACTOR 5
void neural::scaleinputs()
{
	for (int i = 0; i<INPUTS; i++)
	{
		/// if an input is outside of the expected bounds, make it one of the bounds.
		/// BREAKPOINT note that this could hide errors in other calculations
		input[i] = fmin(input[i], maxinput[i]);
		input[i] = fmax(input[i], mininput[i]);

		scaledinput[i] = (input[i] - mininput[i]) / (maxinput[i] - mininput[i]);
		//cout << "scaled: " << scaledinput[i] << endl;
	}
}

/// Neural Net testing function (set to x^2+1)
void neural::score()
{
	error = output[0] - (input[0] * input[0] + 1);
	fitness = 100 - fabs(error);
	static int count;
	count++;
	//if(count%100==0){cout << " &&*  " << scaledinput[0] << " "<< outputout[0] << endl;}
	//cout << "ERR:\t\t" << error << endl;
	//cout << "FITNESS: " << fitness << endl;
	///cout << inputs[0] << "^2+1=" << output[0]<<endl;
}

void neural::ranker(neural* net)
{
	for (int i = 0; i<EVOPOP; i++)
	{
		net[i].rank = 98; if (DEBUG)cout << "net[" << i << "].rank=" << net[i].rank << endl;
	}

	double temp = -1000;
	double max = -1000;
	int best = -1;

	double line = 100000;

	if (DEBUG == 1){ cout << "RANKER 1" << endl; }
	for (int j = 0; j<EVOPOP; j++)
	{
		best = -1;
		temp = -1000000000;
		max = -1000000000;
		for (int i = 0; i<EVOPOP; i++)
		{
			temp = net[i].fitness;
			if (DEBUG == 1){ cout << "RANKER 2 " << i << " " << net[i].fitness << " \tmax: " << max << "\t" << temp << "\tRank:" << net[i].rank << endl; }
			if (temp>max && temp<line)// net[i].rank==98)
			{
				max = temp;
				best = i; if (DEBUG){ cout << "BEST:  " << best << endl; }
			}
			if (DEBUG == 1){ cout << "RANKER 3  " << best << endl; }
		}
		//net[best].rank=EVOPOP-j-1;
		if (DEBUG == 1){ cout << "RANKER 3.5 " << net[best].rank << " best " << best << endl; }
		if (DEBUG == 1){ cout << "RANKER 3.6 " << net[best].rank << endl; }

		if (best == -1)
		{
			for (int kk = 0; kk<EVOPOP; kk++)
			{
				if (DEBUG == 1){ cout << "1" << endl; }
				if (net[kk].rank == -1)
				{
					net[kk].rank = 99;
				}
			}
		}
		else{
			net[best].rank = j;
			line = net[best].fitness;
		}

		if (DEBUG == 1){ cout << "RANKER 3.7 " << net[best].rank << endl; }
		if (DEBUG == 1){ cout << "RANKER 4 " << net[best].rank << endl; }
	}
	if (DEBUG == 1){ cout << "RANKER 5" << endl; }
}

void neural::sorter(neural* net)
{
	for (int j = 0; j<EVOPOP; j++)
	{
		for (int i = 0; i<EVOPOP; i++)
		{
			swap(net, i, net[i].rank);
		}
	}
	if (COBALT>100)
	{
		for (int i = 0; i<EVOPOP; i++)
		{
			//cout << "DOUBLE CHECKERS\t" << i << "\t" << net[i].rank << "\t" << net[i].fitness << endl;
		}
	}
}

void neural::swap(neural* net, int from, int to)
{
	double temporary = 0;
	for (int h = 0; h<HIDDEN; h++)
	{
		temporary = net[from].biasweightIH[h];
		net[from].biasweightIH[h] = net[to].biasweightIH[h];
		net[to].biasweightIH[h] = temporary;

		for (int i = 0; i<INPUTS; i++)
		{
			temporary = net[from].weightIH[i][h];
			net[from].weightIH[i][h] = net[to].weightIH[i][h];
			net[to].weightIH[i][h] = temporary;
		}
		for (int o = 0; o<OUTPUTS; o++)
		{
			temporary = net[from].weightHO[h][o];
			net[from].weightHO[h][o] = net[to].weightHO[h][o];
			net[to].weightHO[h][o] = temporary;
		}
	}

	for (int o = 0; o<OUTPUTS; o++)
	{
		temporary = net[from].biasweightHO[o];
		net[from].biasweightHO[o] = net[to].biasweightHO[o];
		net[to].biasweightHO[o] = temporary;
	}

	temporary = net[from].fitness;
	net[from].fitness = net[to].fitness;
	net[to].fitness = temporary;

	temporary = net[from].rank;
	net[from].rank = net[to].rank;
	net[to].rank = temporary;
}

void neural::evolve(neural* net, int z)
{
	double total = 0;
	double avg = 0;
	for (int i = 0; i<EVOPOP; i++)
	{
		total += net[i].fitness;
	}
	avg = total / EVOPOP;

	int successful;
	successful = LYRAND*SURVIVORPOP;

	if (z>SURVIVORPOP)
	{
		for (int in = 0; in<INPUTS; in++)
		{
			for (int hid = 0; hid<HIDDEN; hid++)
			{
				weightIH[in][hid] = net[successful].weightIH[in][hid] + LYRAND*weightstep;
				weightIH[in][hid] -= LYRAND*weightstep; /// TODO MAKEBETTER
			}
		}

		for (int hid = 0; hid<HIDDEN; hid++)
		{
			biasweightIH[hid] = net[successful].biasweightIH[hid] + LYRAND*weightstep;
			biasweightIH[hid] -= LYRAND*weightstep; /// TODO MAKEBETTER

			for (int out = 0; out<OUTPUTS; out++)
			{
				weightHO[hid][out] = net[successful].weightHO[hid][out] + LYRAND*weightstep;
				weightHO[hid][out] -= LYRAND*weightstep; /// TODO MAKEBETTER
			}
		}

		for (int out = 0; out<OUTPUTS; out++)
		{
			biasweightHO[out] = net[successful].biasweightHO[out] + LYRAND*weightstep;
			biasweightHO[out] -= LYRAND*weightstep; /// TODO MAKEBETTER
		}
	}
	//cout << "SUC\t" << successful << "\n\n";
	//changeweightstep();
}

void neural::scaleoutputs()
{
	for (int o = 0; o<OUTPUTS; o++)
	{
		output[o] = outputout[o] * (maxoutput[o] - minoutput[o]);
		output[o] = output[o] + minoutput[o];
	}
	//cout << "SPEEDY "<< output[0] << endl;
}

double neural::start()
{
	taken = -1;
	fitness = -1;
	rank = -1;
	weightstep = -1;
	for (int i = 0; i<INPUTS; i++)
	{
		input[i] = -1;
		hiddenin[i] = -1;
		scaledinput[i] = -1;

		for (int h = 0; h<HIDDEN; h++)
		{
			weightIH[i][h] = -1;
		}
	}
	for (int h = 0; h<HIDDEN; h++)
	{
		biasweightIH[h] = -1;
		hiddenout[h] = -1;
		outputin[h] = -1;
		for (int o = 0; o<OUTPUTS; o++)
		{
			weightHO[h][o] = -1;
		}
	}
	for (int o = 0; o<OUTPUTS; o++)
	{
		biasweightHO[o] = -1;
		output[o] = -1;
		outputout[o] = -1;
	}
	return 0;
}

void neural::take_limits(vector<double> mini, vector<double> maxi, vector<double> mino, vector<double> maxo)
{
	//cout << "preout" << endl;
	for (int o = 0; o<OUTPUTS; o++)
	{
		//cout << "o=" << o << endl;
		//cout << "maxo=" << maxo.at(o) << endl;
		//cout << "mino=" << mino.at(o) << endl;
		//cout << "maxoutput[o]=" <<maxoutput[o] << endl;
		//cout << "minoutput[o]=" << minoutput[o] << endl;
		maxoutput[o] = maxo.at(o);
		minoutput[o] = mino.at(o);
	}

	//cout << "prein" << endl;
	for (int i = 0; i<INPUTS; i++)
	{
		maxinput[i] = maxi.at(i);
		mininput[i] = mini.at(i);
	}

}

double neural::initialize()
{
	fitness = 0; /// TODO SOMETIMES -10000
	rank = -1;
	weightstep = STARTWEIGHTSTEP;
	taken = 0;
	for (int i = 0; i<INPUTS; i++)
	{
		input[i] = 0;
		hiddenin[i] = 0;
		scaledinput[i] = 0;

		for (int h = 0; h<HIDDEN; h++)
		{
			weightIH[i][h] = LYRAND*SMALLISH-LYRAND*SMALLISH;//(double)rand()/RAND_MAX*SMALLISH;
		}
	}
	for (int h = 0; h<HIDDEN; h++)
	{
		biasweightIH[h] = LYRAND*SMALLISH-LYRAND*SMALLISH;//(double)rand()/RAND_MAX*SMALLISH;
		hiddenout[h] = 0;
		outputin[h] = 0;
		for (int o = 0; o<OUTPUTS; o++)
		{
			weightHO[h][o] = LYRAND*SMALLISH-LYRAND*SMALLISH;//(double)rand()/RAND_MAX*SMALLISH;
		}
	}
	for (int o = 0; o<OUTPUTS; o++)
	{
		biasweightHO[o] = LYRAND*SMALLISH-LYRAND*SMALLISH;//(double)rand()/RAND_MAX*SMALLISH;
		outputout[o] = 0;
		output[o] = 0;
	}
	//take_limits();

#define READLIMITS
	//#define WRITELIMITS


	///INPUT AND OUTPUT LIMITS BLOCK
#ifdef READLIMITS
#endif

#ifdef WRITELIMITS
	cout << "writing limits" << endl;
	cout << "Limit 1 Min: ";//' << endl;
	cin >> a;
#endif
	///INPUT AND OUTPUT LIMITS BLOCK

	return 0;
}

double neural::reset(){
    fitness=0;
}

double neural::go()
{
	scaleinputs();
	for (int i = 0; i<HIDDEN; i++)
	{
		inputtohidden(i);
	}
	for (int j = 0; j<OUTPUTS; j++)
	{
		hiddentoout(j);
	}
	scaleoutputs();
	//score();
	//printresult();
	return 0;
}

/// C1
/// TODO PUT IN SCALING
double neural::inputtohidden(int h)
{

	for (int i = 0; i<INPUTS; i++)
	{
		hiddenin[i] = scaledinput[i]; /// TODO NOTE TO SELF IN OTHER VERSIONS OF THIS CODEOLOGY, THIS MAY BE INCORRECT (INPUTS)
	}
	double tot = 0;
	for (int i = 0; i<INPUTS; i++)
	{
		tot += hiddenin[i] * weightIH[i][h];
		tot += biasweightIH[h];
	}
	hiddenout[h] = sigmoid(tot);

	//cout << "B4HID::" << h << ":::" << tot << endl;
	//cout << "HIDDEN " << h << "::" << hiddenout[h] << endl;

	return hiddenout[h];
}

/// C1
double neural::hiddentoout(int o)
{
	for (int h = 0; h<HIDDEN; h++)
	{
		outputin[h] = hiddenout[h];
	}
	double tot = 0;
	for (int h = 0; h<HIDDEN; h++)
	{
		tot += outputin[h] * weightHO[h][o];
		biasweightHO[o] = 0;
		tot += biasweightHO[o];
	}
	outputout[o] = sigmoid(tot);
	return outputout[o];
}

///C1
double neural::sigmoid(double input)
{
	double output = 0;
	output = 1 / (1 + exp(-input));
	//if(input>=2){output=1;}
	//else if(input < 2 && input > -2){output=input/2;}
	return output;
}

/*

int main()
{

cout << "Pincer 1" << endl;

srand(time(NULL));

FILE * pFile;
neural* pTest;
pTest=new neural[EVOPOP];
pFile = fopen ("ERROR.txt","w");

for(int i=0; i<EVOPOP; i++)
{
pTest[i].start();
pTest[i].initialize();
}

/// INPUT AND OUTPUT LIMIT BLOCK
for(int i=0; i<EVOPOP; i++)
{
pTest[i].mininput[0]=0;
pTest[i].maxinput[0]=5;

pTest[i].minoutput[0]=0;
pTest[i].maxoutput[0]=26;
}
/// INPUT AND OUTPUT LIMIT BLOCK

int num=0;

for(int k=0; k<EPOCHS; k++)
{
for(int i=0; i<EVOPOP; i++)
{pTest[i].fitness=0;}

for(int j=0; j<STEPS; j++)
{
num=j;
cout << "\t\t\t\tnumg:" << num << endl;
if(num>5){num=0;}
for(int i=0; i<EVOPOP; i++)
{
pTest[i].readinputs(num);
//pTest[i].readinputs(4);
}
//cout << "Pincer 1" << endl;

for(int i=0; i<EVOPOP; i++)
{
//pTest[i].scaleinputs();
pTest[i].go();
//pTest[i].scaleoutputs();
//pTest[i].score();
//pTest[0].ranker(pTest);
//pTest[0].sorter(pTest);
//pTest[i].evolve(pTest,i);
//cout << "STEPS: " << j << " , " << k << endl;
}

pTest[0].ranker(pTest);
pTest[0].sorter(pTest);

for(int i=0; i<EVOPOP; i++)
{
pTest[i].evolve(pTest,i);
}

fprintf(pFile,"%.2f\n",pTest[0].fitness);
}
//cout << "EPOCH: " << k << endl;
}

/// TRAIN
/// //////////
/// TEST

for(int i=0; i<EVOPOP; i++)
{pTest[i].fitness=0;}

for(int i=0; i<EVOPOP; i++)
{
pTest[i].readinputs(4);
}

for(int i=0; i<EVOPOP; i++)
{
//pTest[i].scaleinputs();
pTest[i].go();
//pTest[i].scaleoutputs();
//pTest[i].score();
pTest[i].printresult();

//pTest[i].
//cout << "STEPS: " << j << " , " << k << endl;
}


fclose (pFile);
cout << "START" << endl;
return 0;
}
*/


#endif /* NNLIB_H_ */