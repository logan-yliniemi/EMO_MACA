
/*
* File:   main.cpp
* Author: ylinieml
*
* Created on November 5, 2012, 2:15 PM
*/

#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>
#include <assert.h>

using namespace std;

#define ARCHIVE_SIZE 100
#define POPULATION_SIZE POPULATION
#define SPEA_DEBUG 0
#define SPEA_DEBUG_ES 0
#define SPEA_DEBUG_SPECIAL 0

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

class SPEA_2_mem{
public:
	vector<double> coordinates;
	int index;
	double final_fitness;
	int raw_fitness;
	int strength;
	vector<int> dominated_by;
	vector<int> dominates;
	vector<double> distances;
	double raw_distance_indicator;
	double distance_indicator;
	void startup();

	void take_agent(Evo_Agent_KUR);
	Evo_Agent_KUR agent;
};

void SPEA_2_mem::take_agent(Evo_Agent_KUR a){
	agent = a;
}

void SPEA_2_mem::startup(){

}

bool sort_spea_first_dim(SPEA_2_mem const &x, SPEA_2_mem const &y)
{
	return x.coordinates.at(0) < y.coordinates.at(0);
}
bool sort_spea_second_dim(SPEA_2_mem const &x, SPEA_2_mem const &y)
{
	return x.coordinates.at(1) < y.coordinates.at(1);
}
bool sort_spea_third_dim(SPEA_2_mem const &x, SPEA_2_mem const &y)
{
	return x.coordinates.at(2) < y.coordinates.at(2);
}
bool sort_spea_fitness(SPEA_2_mem const &x, SPEA_2_mem const &y)
{
	return x.final_fitness < y.final_fitness;
}

class SPEA_2 {
private:
public:
	void find_knn_dist(int);
	void find_strengths();
	void find_raw_fitness();
	void calc_fitness();
	bool first_dominates_second(vector<double>, vector<double>);
	void clean_archive();
	vector< SPEA_2_mem > population;
	vector< SPEA_2_mem > archive;
	vector< SPEA_2_mem > superset;


	void vector_input(vector<double>, int);
	void two_dim_input(double, double, int);
	void three_dim_input(double, double, double, int);
	void SPEA_reset();
	void size_check();
	void execute(vector<int>*);
	void get_SPEA_fitness(vector<double>*);
	void look();
	void develop_superset();
	void environmental_selection();
	void truncate(vector<SPEA_2_mem>*);
	vector<int> mating_selection();
	void take_agent(Evo_Agent_KUR, int);
	void final_archive();
};

void SPEA_2::final_archive(){
	FILE* pFILE;
	pFILE = fopen("spea_final_archive.txt", "w");
	cout << "archive size: " << archive.size() << endl;
	for (int i = 0; i<archive.size(); i++){
		report(pFILE, archive.at(i).coordinates.at(0), 1);
		report(pFILE, archive.at(i).coordinates.at(1), 0);
		newline(pFILE);
	}
	fclose(pFILE);
}

void SPEA_2::clean_archive(){
	for (int i = 0; i<archive.size(); i++){
		archive.at(i).dominated_by.clear();
		archive.at(i).dominates.clear();
		//archive.at(i).
	}
}

void SPEA_2::take_agent(Evo_Agent_KUR A, int spot){
	population.at(spot).take_agent(A);
}

void SPEA_2::environmental_selection(){
	if (SPEA_DEBUG_ES){ cout << "ENV SELECTION IN" << endl; }

	/// sort by fitness
	vector<SPEA_2_mem>* pS = &superset;
	sort(superset.begin(), superset.end(), sort_spea_fitness);

	/// move all nondominated solutions into the next archive
	vector<SPEA_2_mem> next_arch;
	vector<SPEA_2_mem>* pNA = &next_arch;
	int z = 0;
	if (SPEA_DEBUG_ES){
		cout << "SUPERSET DOMINATED BY SIZES" << endl;
	}
	for (int i = 0; i<superset.size(); i++){
		if (SPEA_DEBUG_ES){
			cout << superset.at(i).dominated_by.size() << "\t";
			cout << superset.at(i).coordinates.at(0) << "\t";
			cout << superset.at(i).coordinates.at(1) << "\n";
			//cout << superset.at(i).coordinates.at(2) << "\n";
		}
		if (superset.at(i).dominated_by.size() == 0){
			next_arch.push_back(superset.at(i));
			z = i;
		}
	}

	if (SPEA_DEBUG_ES){ cout << "Next arch before size adjustment \t" << next_arch.size() << endl; }
	/// if this is less than the archive size, keep going, grab solutions up to the size.
	while (next_arch.size() < ARCHIVE_SIZE){
		if (z>superset.size() - 1){ break; }
		next_arch.push_back(superset.at(z));
		//cout << superset.at(z).agent.actions.size() << "\t\tXX" << endl;
		z++;
	}
	if (SPEA_DEBUG_ES){ cout << "Next arch while possibly above limit \t" << next_arch.size() << endl; }
	/// if this is more than the archive size, we truncate.
	while (next_arch.size() > ARCHIVE_SIZE){
		truncate(pNA);
	}
	if (SPEA_DEBUG_ES){ cout << "Next archive after all \t" << next_arch.size() << endl; }

	archive = next_arch;
}

bool SPEA_2::first_dominates_second(vector<double> a, vector<double> b){
	bool adomb = true;
	if (a == b){
		return false;
	}

	for (int i = 0; i<a.size(); i++){
		if (a.at(i) < b.at(i)){
			adomb = false;
			return adomb;
		}
	}

	return adomb;
}

void SPEA_2::SPEA_reset() {

}

void SPEA_2::vector_input(vector<double> put_in, int dex) {
	//cout << "vector_input is not yet developed. This run is void." << endl;
	//vector<double> all_inputs;
	SPEA_2_mem mem;
	mem.startup();
	mem.coordinates = put_in;
	mem.index = dex;
	population.push_back(mem);
}

void SPEA_2::two_dim_input(double a, double b, int dex) {
	vector<double> put_in;
	put_in.push_back(a);
	put_in.push_back(b);

	SPEA_2_mem mem;
	mem.startup();
	mem.coordinates = put_in;
	mem.index = dex;
	population.push_back(mem);
}

void SPEA_2::three_dim_input(double a, double b, double c, int dex) {
	vector<double> put_in;
	put_in.push_back(a);
	put_in.push_back(b);
	put_in.push_back(c);

	SPEA_2_mem mem;
	mem.startup();
	mem.coordinates = put_in;
	mem.index = dex;

	population.push_back(mem);
}

void SPEA_2::truncate(vector<SPEA_2_mem>* pNA){
	if (SPEA_DEBUG_SPECIAL){ cout << "Truncating: " << pNA->size(); }
	int num_pts = pNA->size();

	double knnforcout;

	for (int i = 0; i<num_pts; i++){
		pNA->at(i).distances.clear();
		pNA->at(i).distances.resize(num_pts);
	}

	for (int i = 0; i < num_pts; i++){
		for (int j = i; j<num_pts; j++){
			double ijdist = vectordist(
				superset.at(i).coordinates,
				superset.at(j).coordinates);
			pNA->at(i).distances.at(j) = ijdist;
			pNA->at(j).distances.at(i) = ijdist;
			assert(pNA->at(i).distances.size() == num_pts);
		}
	}
	int cull = -1;
	double minn = 999999999999;
	for (int i = 0; i<num_pts; i++){
		sort(pNA->at(i).distances.begin(), pNA->at(i).distances.end());
		double knn = pNA->at(i).distances.at(10);
		if (knn<minn){
			minn = knn;
			cull = i;
			knnforcout = knn;
		}
	}
	if (SPEA_DEBUG_SPECIAL){
		cout << "\tculling knn dist of " << knnforcout << endl;
	}
	pNA->erase(pNA->begin() + cull);
}

void SPEA_2::find_knn_dist(int kth) {
	int num_pts = superset.size();
	// cout << "inside knn_dist" << endl;
	// cout << superset.size() << endl;

	for (int i = 0; i<num_pts; i++){
		superset.at(i).distances.clear();
		superset.at(i).distances.resize(num_pts);
	}

	for (int i = 0; i < num_pts; i++){
		for (int j = i; j<num_pts; j++){
			double ijdist = vectordist(superset.at(i).coordinates, superset.at(j).coordinates);
			superset.at(i).distances.at(j) = ijdist;
			superset.at(j).distances.at(i) = ijdist;
		}
	}
	for (int i = 0; i<num_pts; i++){
		sort(superset.at(i).distances.begin(), superset.at(i).distances.end());
		double knn = superset.at(i).distances.at(kth);
		if (SPEA_DEBUG){
			for (int j = 0; j<kth; j++){
				cout << i << "'s knn " << j << " = " << superset.at(i).distances.at(j) << endl;
			}
		}
		superset.at(i).distance_indicator = 1 / (knn + 2);
	}
}

void SPEA_2::develop_superset() {
	superset.clear();
	superset.reserve(population.size() + archive.size());
	superset.insert(superset.end(), population.begin(), population.end());
	superset.insert(superset.end(), archive.begin(), archive.end());
}

void SPEA_2::find_strengths() {
	for (int i = 0; i< superset.size(); i++){
		superset.at(i).dominates.clear();
		superset.at(i).dominated_by.clear();
	}

	for (int i = 0; i< superset.size(); i++){
		for (int j = 0; j<superset.size(); j++){
			if (i == j){ continue; }
			if (first_dominates_second(superset.at(i).coordinates, superset.at(j).coordinates)){
				/// i dominates j
				superset.at(i).dominates.push_back(j);
				superset.at(j).dominated_by.push_back(i);
			}
		}
	}

	if (SPEA_DEBUG){
		for (int i = 0; i<superset.size(); i++){
			cout << "SUPERSET " << i << " DOMINATED BY VECTOR: " << endl;
			for (int j = 0; j<superset.at(i).dominated_by.size(); j++){
				cout << superset.at(i).dominated_by.at(j) << "\t";
			}
			cout << endl;
		}
	}

	for (int i = 0; i< superset.size(); i++){
		superset.at(i).strength = superset.at(i).dominates.size();
	}
}

void SPEA_2::find_raw_fitness() {
	for (int i = 0; i<superset.size(); i++){
		superset.at(i).raw_fitness = 0;
	}
	for (int i = 0; i<superset.size(); i++){
		if (SPEA_DEBUG){
			cout << "\tSuperset member " << i << " is dominated by " << superset.at(i).dominated_by.size() << " individuals." << endl;
		}
		for (int z = 0; z<superset.at(i).dominated_by.size(); z++){
			int idomby = superset.at(i).dominated_by.at(z);
			if (SPEA_DEBUG){
				cout << "\t " << i << " is dominated by " << superset.at(i).dominated_by.at(z) << endl;
				cout << "\t at a score of " << superset.at(i).coordinates.at(0) << "," << superset.at(i).coordinates.at(1) << endl;
				cout << " to " << superset.at(idomby).coordinates.at(0) << "," << superset.at(idomby).coordinates.at(1) << endl;
				cout << "\t member " << idomby << " has a strength of " << superset.at(idomby).strength << endl;
			}
			superset.at(i).raw_fitness += superset.at(idomby).strength;
			if (SPEA_DEBUG){
				cout << "\t making " << i << "'s raw fitness cumulative " << superset.at(i).raw_fitness << endl << endl;
			}
		}
	}
}

void SPEA_2::calc_fitness() {
	for (int i = 0; i<superset.size(); i++){
		superset.at(i).final_fitness = superset.at(i).raw_fitness + superset.at(i).distance_indicator;
	}

	if (SPEA_DEBUG){
		cout << "CALC FITNESS WATCHER: (x,y,domsize,distindic,raw,final) " << endl;
		for (int i = 0; i<superset.size(); i++){
			cout << superset.at(i).coordinates.at(0) << "\t";
			cout << superset.at(i).coordinates.at(1) << "\t";
			cout << superset.at(i).dominated_by.size() << "\t";
			cout << superset.at(i).distance_indicator << "\t";
			cout << superset.at(i).raw_fitness << "\t";
			cout << superset.at(i).final_fitness << endl;
		}
	}
}

void SPEA_2::size_check() {

}

void SPEA_2::look() {

}

void SPEA_2::execute(vector<int>* pV) {
	if (SPEA_DEBUG) { cout << "SPEA 2 CALLED" << endl; }
	clean_archive();

	develop_superset();

	find_strengths();
	//size_check();

	find_raw_fitness();
	//size_check();

	find_knn_dist(10);
	//size_check();

	calc_fitness();
	//size_check();
	environmental_selection();

	vector<int> a = mating_selection();
	for (int i = 0; i<a.size(); i++){
		pV->push_back(a.at(i));
	}

	population.clear();
}

vector<int> SPEA_2::mating_selection(){
	/// binary tournament search with replacement.
	vector<int> reproduce;
	/// select two:
	for (int i = 0; i<POPULATION_SIZE; i++){
		int a = rand() % archive.size();
		int b = rand() % archive.size();
		/// compare
		if (archive.at(a).final_fitness < archive.at(b).final_fitness){
			/// a wins
			reproduce.push_back(a);
		}
		if (archive.at(b).final_fitness < archive.at(a).final_fitness){
			/// b wins
			reproduce.push_back(b);
		}
		if (archive.at(a).final_fitness == archive.at(b).final_fitness){
			i--;
		}
	}
	if (SPEA_DEBUG){ cout << "REPRODUCE SIZE: " << reproduce.size() << endl; }
	return reproduce;
}

void SPEA_2::get_SPEA_fitness(vector<double>* pFitness) {

}
