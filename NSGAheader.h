#ifndef NSGAHEADER_H_INCLUDED
#define NSGAHEADER_H_INCLUDED

/*
* File:   main.cpp
* Author: ylinieml
*
* Created on December 12, 2012, 11:30 AM
*/

#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>
#include <algorithm>

#define NSGA_DEBUG 0

using namespace std;

class NSGA_mem{
public:
	vector<double> coordinates;
	int index;
	double final_fitness;
	int pareto_tier;
	double raw_distance_indicator;
	double distance_indicator;
};

bool sort_nsga_first_dim(NSGA_mem const &x, NSGA_mem const &y)
{
	return x.coordinates.at(0) < y.coordinates.at(0);
}
bool sort_nsga_second_dim(NSGA_mem const &x, NSGA_mem const &y)
{
	return x.coordinates.at(1) < y.coordinates.at(1);
}
bool sort_nsga_third_dim(NSGA_mem const &x, NSGA_mem const &y)
{
	return x.coordinates.at(2) < y.coordinates.at(2);
}
bool sort_nsga_fitness(NSGA_mem const &x, NSGA_mem const &y)
{
	return x.final_fitness < y.final_fitness;
}

//bool SortByDim(NSGA_mem* left, NSGA_mem* right, int dim) {
//        int al = left->coordinates.at(dim), ar = right->coordinates.at(dim);
//        return (al < ar);
//}

class NSGA_2 {
private:
public:
	vector<NSGA_mem> population;
	int NSGA_dimension;
	void find_tier();
	void find_proximity(vector<NSGA_mem>* pM);
	void interpret();

	void vector_input(vector<double>, int);
	void two_dim_input(double, double, int);
	void three_dim_input(double, double, double, int);

	void NSGA_reset();
	void size_check();
	void execute();
	void get_NSGA_fitness(vector<double>*);
	double NSGA_member_fitness(int);
	void look();
	void declare_NSGA_dimension(int);
    void membersort(vector<NSGA_mem>* pM, int);

	void show_tiers();
	bool first_dominates_second(vector<double> a, vector<double> b);
};

void NSGA_2::declare_NSGA_dimension(int a)
{
	if (NSGA_DEBUG){ cout << "NSGA::dimension in " << a << endl; }
	NSGA_dimension = a;
}

void NSGA_2::NSGA_reset() {
	if (NSGA_DEBUG){ cout << "NSGA::reset in" << endl; }
	population.clear();
}

void NSGA_2::vector_input(vector<double> inputs, int dex) {
	NSGA_mem ind;
	ind.index = dex;
	ind.pareto_tier = -1;
	population.push_back(ind);

	population.at(dex).coordinates = inputs;
}

void NSGA_2::two_dim_input(double a, double b, int dex) {
	if (NSGA_DEBUG){ cout << "NSGA::input in " << a << "," << b << endl; }
	NSGA_mem ind;
	ind.index = dex;
	ind.pareto_tier = -1;
	population.push_back(ind);

	vector<double> put_in;
	put_in.push_back(a);
	put_in.push_back(b);
	population.at(dex).coordinates = put_in;
}

void NSGA_2::three_dim_input(double a, double b, double c, int dex) {
	NSGA_mem ind;
	ind.index = dex;
	ind.pareto_tier = -1;
	population.push_back(ind);

	vector<double> put_in;
	put_in.push_back(a);
	put_in.push_back(b);
	put_in.push_back(c);
	population.at(dex).coordinates = put_in;
}

#ifndef LYVECTORDIST
#define LYVECTORDIST
double vectordist(vector<double> a, vector<double> b){
	double sq = 0;
	double del;
	for (int i = 0; i<a.size(); i++){
		del = a.at(i) - b.at(i);
		sq += (del)*(del);
	}
	return sqrt(sq);
}
#endif

void NSGA_2::membersort(vector<NSGA_mem>* pM, int obj){
    if(obj > pM->at(0).coordinates.size()){
        std::invalid_argument( "NSGA Requested higher dimensional vector than given.");
    }
    
    /// Sort vector of members by objective value, lowest to highest
    for(int z=0; z<pM->size()+1; z++){
    for(int i=0; i<pM->size()-1; i++){
        int j=i+1;
        double val1 = pM->at(i).coordinates.at(obj);
        double val2 = pM->at(j).coordinates.at(obj);
        if(val2 < val1){
            NSGA_mem temp;
            temp = pM->at(i);
            pM->at(i) = pM->at(j);
            pM->at(j) = temp;
        }
    }
    }
}

void NSGA_2::find_proximity(vector<NSGA_mem>* pM) { /// Jan 2014 redux of find_proximity
	if (NSGA_DEBUG){
		cout << "NSGA::proxy in" << endl;
		cout << "PM SIZE: " << pM->size() << endl;
	}

	/// We have brought in a single tier, pM (pointer to Members).
	int DIMS = NSGA_dimension;
	for (int dim = 0; dim<DIMS; dim++){
		/// sort by score.
		for (int i = 0; i<pM->size(); i++){
			pM->at(i).raw_distance_indicator = 0;
		}
		if (dim == 0)
			sort(pM->begin(), pM->end(), sort_nsga_first_dim);
		if (dim == 1)
			sort(pM->begin(), pM->end(), sort_nsga_second_dim);
        if (dim == 2)
            sort(pM->begin(), pM->end(), sort_nsga_third_dim);
        if (dim > 2) membersort(pM,dim);
		if (NSGA_DEBUG){
			for (int i = 0; i<pM->size(); i++){
				cout << "Member " << i << endl;
				cout << "X,Y : " << pM->at(i).coordinates.at(0) << "\t" << pM->at(i).coordinates.at(1) << "\t" << pM->at(i).index << endl;
			}
		}
		/// give minimum a high proxy score.
		pM->at(0).raw_distance_indicator += 1000000;
		/// For each intermediate point, find the point above it.
		for (int i = 1; i<pM->size() - 1; i++){
			pM->at(i).raw_distance_indicator += vectordist(pM->at(i - 1).coordinates, pM->at(i + 1).coordinates);
		}
		/// Also find the nearest point below.
		pM->at(pM->size() - 1).raw_distance_indicator += 1000000;
		/// Do the n-dimensional box for those 2 points
		/// add this to the distance for the point
	}

	/// turn distances into a proximity score.
	// find max raw distance score.
	double maxrawdist = -9999999999999;
	int maxdex = -1;
	for (int i = 0; i<pM->size(); i++){
		double val = pM->at(i).raw_distance_indicator;
		if (val > maxrawdist){
			maxrawdist = val;
			maxdex = i;
		}
	}

	for (int i = 0; i<pM->size(); i++){
		pM->at(i).distance_indicator = pM->at(i).raw_distance_indicator / (maxrawdist + 1.0);
		pM->at(i).final_fitness = pM->at(i).pareto_tier + pM->at(i).distance_indicator;
	}
}

void NSGA_2::interpret()
{
	if (NSGA_DEBUG){ cout << "NSGA::interpret in " << endl; }
	for (int i = 0; i<population.size(); i++){
		cout << population.at(i).pareto_tier << "\t" << population.at(i).raw_distance_indicator << "\t" << population.at(i).distance_indicator << "\t" << population.at(i).final_fitness << endl;
	}

}

void NSGA_2::find_tier() {
	if (NSGA_DEBUG){ cout << "NSGA::tier in " << endl; }
	int tt = 0;
	while (true){
		vector< vector<double> > remaining_coords;
		vector<int> remaining_indexes;
		/// build list of remaining points.
		for (int i = 0; i<population.size(); i++){
			if (population.at(i).pareto_tier == -1){
				remaining_coords.push_back(population.at(i).coordinates);
				remaining_indexes.push_back(i);
			}
		}
		if (remaining_indexes.size() == 0){
			break;
		}
        
		/// of remaining points, find the NDS.
		for (int i = 0; i<remaining_indexes.size(); i++){
			for (int j = 0; j<remaining_indexes.size(); j++){
				if (i == j){ continue; }
				int debug_1 = remaining_coords.size();
				if (first_dominates_second(remaining_coords.at(i), remaining_coords.at(j))){
					remaining_indexes.erase(remaining_indexes.begin() + j);
					remaining_coords.erase(remaining_coords.begin() + j);
					i--;
					j--;
					if (i<0){ i = 0; }
					if (j<0){ j = 0; }
				}
			}
		}

		/// The live ones are part of this tier, tt.
		for (int i = 0; i<remaining_indexes.size(); i++){
			population.at(remaining_indexes.at(i)).pareto_tier = tt;
		}
		tt++;

		vector<NSGA_mem> those_in_tier;
		vector<NSGA_mem>* pT = &those_in_tier;
		for (int i = 0; i<remaining_indexes.size(); i++){
			those_in_tier.push_back(population.at(remaining_indexes.at(i)));
		}
		find_proximity(pT);
		for (int i = 0; i<pT->size(); i++){
			population.at(pT->at(i).index) = pT->at(i);
		}

		/// And we repeat.
	}

	if (NSGA_DEBUG){
		cout << "Tiers: " << endl;
		for (int i = 0; i<population.size(); i++){
			cout << population.at(i).pareto_tier << "\t";
		}
		cout << endl;
	}
}

bool NSGA_2::first_dominates_second(vector<double> a, vector<double> b){
	bool adomb = true;
	for (int i = 0; i<a.size(); i++){
		if (a.at(i) < b.at(i)){
			adomb = false;
			break;
		}
	}
	return adomb;
}

void NSGA_2::size_check() {

}

void NSGA_2::execute() {
	if (NSGA_DEBUG){ cout << "NSGA::EXECUTE in " << endl; }
	find_tier();
}

void NSGA_2::get_NSGA_fitness(vector<double>* pFitness) {
	if (NSGA_DEBUG){ cout << "NSGA::getfitness in " << endl; }
	pFitness->clear();
	/// LYLY
}
double NSGA_2::NSGA_member_fitness(int dex){
	return population.at(dex).final_fitness;
}

void NSGA_2::show_tiers()
{

}

/// INSTRUCTIONS:
/*
* NSGA_2 NSGA;
* NSGA.declare_NSGA_dimension(2);
* NSGA.NSGA_reset();
* <<<<INPUTS GO IN HERE, probably NSGA.two_dim_input(a,b);>>>>>
* NSGA.execute();
* vector<double> fitnesses;
* vector<double>* pFit=&fitnesses;
* NSGA.get_NSGA_fitness(pFit);
*/

#endif // NSGAHEADER_H_INCLUDED
