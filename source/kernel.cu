//////////////CUDA INCLUDES///////////////
#include "cuda.h"
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"

//////////////////////////////////////////

#include "stdafx.h"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <windows.h>
#include <limits>
#include <math.h>
#include <vector>
#include <list>
#include <time.h>
#include <algorithm>
#include <assert.h>

#undef max
#undef min

#define M_PI            3.14159265358979323846  /* pi */
#define maxOf(a, b) (((a) > (b)) ? a : b)

using namespace std;

struct Neuron;
typedef Neuron *ptNeuron;

struct arc {
	ptNeuron target = nullptr;
	float weight = 0;
	float oldDelta = 0;
	//bool enabled = true;
};

struct interArc {
	ptNeuron target = nullptr;
	ptNeuron base = nullptr;
};

struct Neuron {
	vector<arc> OutArcs; // to inizialize: = new vector<arc>(10);
	u_int numOutArcs = 0; // numero di archi in uscita
	u_int numInArcs = 0; //numero di archi in ingresso
	u_int layer = 0; //indice riga del neurone
	u_int column = 0; //indice della colonna del neurone
	float bayes = 0.01f; //peso del bayes
	float oldBayesDelta = 0; //ultima variazione del peso
	//vector<float> timeBayes; // vettore delle interconnessioni temporali
	vector<float> influenceInput; // vettore contenente la percentuale di influenza relativa ad ogni input
	vector<float> influenceOutput; // vettore contenente la percentuale di influenza relativa all'errore retropropagato da ogni output
	float output = 0; //potenziale attuale del neurone
	float absOutSum = 0; //somma in valore assoluto degli input del neurone
	float absDeltaSum = 0; //somma in valore assoluto delle variazioni dei pesi
	float BPerr = 0; //errore di retropropagazione
	int neurIdx = 0; //ogni neurone è contraddistinto da un indice unico che si riferisce alla sua posizione
};

struct Layer {
	vector<Neuron> Neurons;
	u_int numNeurons = 0;
};

struct Omap {
	float maxValue = 1;
	float minValue = 0;
};

struct conMap { // struttura necessaria per l'inserimento di un nuovo neurone
	u_int startLyr;
	u_int startCol;
	u_int arcRef;
	u_int targetLyr;
	u_int targetCol;
};

struct timeSeries {
	list<float> evento;
};

struct example {
	vector<float> input;
	vector<float> Doutput;
};

struct Dataset {
	vector<example> trainingSet;
	vector<example> validationSet;
	float triningErr = 0;
	float validationErr = 0;
};

class DatasetCore {
public:
	list<Dataset> Datasets;

	DatasetCore() {

	}
	////////////////////////////////////////////////MANIPOLAZIONE DATASET//////////////////////////////////////
	void readTimeSeriesCsv(string filename, int outStep, int inEx, float trainingPerc) {
		//outStep - rappresenta il numero di valori per esempio output
		//inEx - è il numero di esempi output precedenti che vengono passati in ogni esempio input
		//trainingPerc - è la percentuale di dataset da usare come trainingset
		cout << "caricamento del file " << filename << endl;
		ifstream file(filename + ".csv");
		stringstream stream;
		stream << file.rdbuf();
		string line, cell;
		int col = 0, row = -1;
		int fileRow = coutnRow(filename);
		list<timeSeries> esempi(fileRow);
		list<timeSeries>::iterator it = esempi.begin();

		while (getline(stream, line, '\n')) { //smista righe
			stringstream streamline(line);
			col = 0; row++;
			it->evento.resize(0);
			while (getline(streamline, cell, ';')) {
				//cout << stof(cell) << endl;
				it->evento.push_back(stof(cell));
			}
			it++;
		}

		Dataset newSet;
		example newEx;
		newEx.Doutput.resize(outStep);
		newEx.input.resize(inEx*outStep);
		int nEx = esempi.size() - inEx;
		int ntrainEx = (int)((trainingPerc / 100) * nEx);
		int nvalidateEx = nEx - ntrainEx;
		int sPos = 0, ePos = inEx, pos = 0; // il passo è outStep
		int inId = 0, outId = 0;
		for (int i = 0; i < nEx; i++) {
			pos = 0, inId = 0, outId = 0;

			for (list<timeSeries>::iterator p = esempi.begin(); p != esempi.end(); p++) {
				for (list<float>::iterator q = p->evento.begin(); q != p->evento.end(); q++) {
					if (pos >= sPos && pos <= ePos) {
						if (pos >= inEx + sPos) { // carico il vettore output
							newEx.Doutput[outId] = *q;
							outId++;
						}
						else { //carico il vettore di input
							newEx.input[inId] = *q;
							inId++;
						}
					}
				}
				pos++;
			}

			if (sPos < ntrainEx) { // carico il trining set
				newSet.trainingSet.push_back(newEx);
			}
			else { //carico il validationset
				newSet.validationSet.push_back(newEx);
			}
			//cout << sPos << endl;
			sPos++;
			ePos++;
		}
		Datasets.push_back(newSet);
	}
	vector<example> getDataset(int n, bool training = true) {
		list<Dataset>::iterator p = Datasets.begin();
		for (int k = 0; k < n; k++) p++;
		if (training == true) {
			return p->trainingSet;
		}
		else {
			return p->validationSet;
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////ALTRE FUNZIONI//////////////////////////////////////////////
	int coutnRow(string filename) {
		ifstream file(filename + ".csv");
		stringstream stream;
		stream << file.rdbuf();
		string line;
		int row = 0;
		while (getline(stream, line, '\n'))row++;
		return row;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

class Network {

public:
	vector<Layer> Layers; // vettore di struct layer
	vector<example> examples; // vettore di esempi per l'apprendimento
	vector<Omap> map; // vettore contenente i valori di rimappatura del'output della rete

	string genoma = ""; // nome del file
	u_int nLayers = 0; // Layer nella rete compresi input output
	u_int nNeurons = 0; // numero totale neuroni nella rete
	u_int nArc = 0; //numero totale di tutti gli archi presenti nella rete

	Network(string filename) {
		genoma = filename;
	}

	////////////////////////////////////FUNZIONE COSTRUZIONE RETE DA FILE/////////////////////////////////////////////////////////
	void getNetParams() {
		ifstream file(genoma + ".txt");
		string line, segment, pice;
		int flag = 4; // 3- NL, 2-Ln, 1-NCON, 0-CON
		int pos;
		int lyr, neuronsInLyr, Nneuron, con; // numero layer, numero neuroni in dato layer, numero connessioni d'uscita in un dato neurone
		int Tlyr, Tneuron; // target layer, target neuron
		int conFlag; // 0 - inserisco la connessione, 1 - inserisco il peso
		float bayes;
		//file.open(filename + ".txt");
		stringstream stream;
		stream << file.rdbuf();

		if (file.is_open()) {
			//cout << "si e' apertpo" << endl;
			while (getline(stream, line, '\n')) { //smista righe
				stringstream streamline(line);
				pos = 0;
				while (getline(streamline, segment, ' ')) { // smista segmenti riga
															//cout << segment << endl;
					if (flag != 0 && pos == 0) {
						if (segment == "CON") {
							flag = 0;
						}
						else if (segment == "NCON") {
							flag = 1;
						}
						else if (segment == "L0") {
							flag = 2;
						}
						else if (segment == "NL") {
							flag = 3;
						}
					}

					if (flag == 0 && segment != "CON") { // dichiarazione delle connessioni
														 //cout << "flag0" << endl;
						if (pos == 0) {
							stringstream streamsegment(segment);
							getline(streamsegment, pice, '-'); // file syntax CON \n 0-0
							lyr = stoi(pice); // rimane fissato per tutta la riga
							getline(streamsegment, pice);
							Nneuron = stoi(pice);
							conFlag = 0;

						}
						else if (lyr != nLayers) {
							if (conFlag == 0) { //inizializzo la connessione

								stringstream streamsegment(segment);
								getline(streamsegment, pice, '-'); // file syntax CON \n 1-0
								Tlyr = stoi(pice);
								getline(streamsegment, pice, ' ');
								Tneuron = stoi(pice);
								Layers[lyr].Neurons[Nneuron].OutArcs[(pos - 1) / 2].target = &(Layers[Tlyr].Neurons[Tneuron]);
								Layers[Tlyr].Neurons[Tneuron].numInArcs++;
								conFlag = 1;

							}
							else { //inizzializzo il peso della connessione 

								Layers[lyr].Neurons[Nneuron].OutArcs[(pos - 2) / 2].weight = stof(segment);
								conFlag = 0;

							}
						}

					}
					else if (flag == 1 && segment != "NCON") { // dichiarazione numero connessioni
															   //cout << "flag1" << endl;
						if (pos == 0) {
							stringstream streamsegment(segment);
							getline(streamsegment, pice, '-'); // file syntax NCON \n 0-0_4
							lyr = stoi(pice);
							getline(streamsegment, pice, '_');
							Nneuron = stoi(pice);
							getline(streamsegment, pice, ':');
							con = stoi(pice);
							getline(streamsegment, pice, ' ');
							bayes = stof(pice);

							if (con >= 0) {
								Layers[lyr].Neurons[Nneuron].numOutArcs = con;
								nArc += con;
								Layers[lyr].Neurons[Nneuron].OutArcs.resize(con, arc());  //= new vector<arc>(con); // dichiarazione numero connessioni del neurone
								Layers[lyr].Neurons[Nneuron].layer = lyr;
								Layers[lyr].Neurons[Nneuron].column = Nneuron;
								Layers[lyr].Neurons[Nneuron].bayes = bayes;
							}
						}
					}
					else if (flag == 2) { // numero neuroni per layer
										  //cout << "flag2" << endl;
						if (pos == 0) {
							lyr = stoi(segment.erase(0, 1));
						}
						else {
							neuronsInLyr = stoi(segment);
							Layers[lyr].numNeurons = neuronsInLyr;
							nNeurons += neuronsInLyr;
							Layers[lyr].Neurons.resize(neuronsInLyr, Neuron());  // = new vector<Neuron>(neuronsInLyr); // dichiarazione numero di neuroni per layer
																				 //cout << segment << " " << lyr << endl;
						}

					}
					else if (flag == 3 && pos != 0) { // numero layer
													  //cout << "flag3" << endl;
						Layers.resize(stoi(segment), Layer()); //= new vector<Layer>(stoi(segment)); // dichiarazione numero di layer
						nLayers = stoi(segment);
						//cout << segment << endl;
					}
					pos++;
				}
			}

			for (int i = 0; i < Layers[nLayers - 1].numNeurons; i++) {
				Layers[nLayers - 1].Neurons[i].column = i;
				Layers[nLayers - 1].Neurons[i].layer = nLayers - 1;
			}

		}
		else {

			cout << genoma + ": errore nell'apertura o file non trovato" << endl;

		}
		file.close();
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////FUNZIONE COSTRUZIONE DATASET DA FILE//////////////////////////////////////////////////////
	void getDataset(string filename) {
		ifstream file(filename + ".txt");
		string line, segment, pice;
		int flag = 4; // 3- NE, 2-IN, 1-OUT, 0-ES
		int pos, esPos;
		int input, output, es; // numero layer, numero neuroni in dato layer, numero connessioni d'uscita in un dato neurone
		int IOFlag = 0; // 0 - input esempio, 1 - output esempio
						//file.open(filename + ".txt");
		stringstream stream;
		stream << file.rdbuf();

		if (file.is_open()) {
			//cout << "si e' apertpo" << endl;
			while (getline(stream, line, '\n')) { //smista righe
				stringstream streamline(line);
				pos = 0;
				while (getline(streamline, segment, ' ')) { // smista segmenti riga
															//cout << segment << endl;
					if (flag != 0 && pos == 0) {
						if (segment == "ES") {
							flag = 0;
						}
						else if (segment == "OUT") {
							flag = 1;
						}
						else if (segment == "IN") {
							flag = 2;
						}
						else if (segment == "NE") {
							flag = 3;
						}
					}
					if (flag == 0 && segment == "ES") { // caricamento degli esmpi
														//cout << "flag0" << endl;
						goto examplesLoading;
					}
					else if (flag == 1 && pos != 0) { // numero output
													  //cout << "flag1" << endl;
						output = stoi(segment);
						if (output != numNeurons(nLayers - 1)) { cout << "Il dataset deve avere lo stesso numero di output della rete!!" << endl; ClearDataset(); return; }
					}
					else if (flag == 2 && pos != 0) { // numero input
													  //cout << "flag2" << endl;
						input = stoi(segment);
						if (input != numNeurons(0)) { cout << "Il dataset deve avere lo stesso numero di input della rete!!" << endl; ClearDataset(); return; }
					}
					else if (flag == 3 && pos != 0) { // numero esempi
													  //cout << "flag3" << endl;
						examples.resize(stoi(segment), example()); //= new vector<Layer>(stoi(segment)); // dichiarazione numero di layer
																   //cout << segment << endl;
					}
					pos++;
				}
			}

		examplesLoading:
			esPos = 0;
			while (getline(stream, line, '\n')) { //smista righe
				stringstream streamline(line);
				pos = 0;
				if (!IOFlag) { // carico gli input
					examples[esPos].input.resize(input);
					while (getline(streamline, segment, ' ')) { // smista segmenti riga
						examples[esPos].input[pos] = stof(segment);
						pos++;
					}
					IOFlag = 1;
				}
				else { // carico gli output
					examples[esPos].Doutput.resize(output);
					while (getline(streamline, segment, ' ')) { // smista segmenti riga
						examples[esPos].Doutput[pos] = stof(segment);
						pos++;
					}
					IOFlag = 0;
					esPos++;
				}

			}

		}
		else {

			cout << genoma + ": errore nell'apertura o file non trovato" << endl;

		}
		file.close();

	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////SALVA RETE SU FILE///////////////////////////////////////////////////////////////
	void saveNet(string filename = "") {

		cout << "saving  net.. " << endl;
		ofstream file;

		if (filename != "") {
			file.open(filename + ".txt");
		}
		else {
			file.open(genoma + ".txt");
		}
		file << "NL " << nLayers << '\n';

		for (int i = 0; i < nLayers; i++) file << "L" << i << " " << Layers[i].numNeurons << '\n';

		file << "NCON" << '\n';
		for (int b = 0; b < nLayers; b++) {
			for (int d = 0; d < Layers[b].numNeurons; d++) {
				file << b << "-" << d << "_" << Layers[b].Neurons[d].numOutArcs << ":" << Layers[b].Neurons[d].bayes << '\n';
			}
		}

		file << "CON";
		for (int b = 0; b < nLayers; b++) {
			//cout << "salvataggio connessioni layer:" << b << endl;
			for (int d = 0; d < Layers[b].numNeurons; d++) {
				file << '\n';
				file << b << "-" << d;

				for (int c = 0; c < Layers[b].Neurons[d].numOutArcs; c++) {
					file << " ";
					file << Layers[b].Neurons[d].OutArcs[c].target->layer << "-" << Layers[b].Neurons[d].OutArcs[c].target->column << " " << Layers[b].Neurons[d].OutArcs[c].weight;
				}
			}
		}
		file.close();
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////SALVA DATASET SU FILE///////////////////////////////////////////////////////////////
	void saveDataset(string filename) {
		ofstream file(filename + ".txt");
		file << "NE " << examples.size() << '\n';
		file << "IN " << Layers[0].numNeurons << '\n';
		file << "OUT " << Layers[nLayers - 1].numNeurons << '\n';
		file << "ES" << '\n';
		for (int i = 0; i < examples.size(); i++) {
			for (int j = 0; j < examples[i].input.size(); j++) { // salvo gli input 
				file << examples[i].input[j];
				if (j == examples[i].input.size() - 1) { file << '\n'; }
				else { file << " "; }
			}
			for (int j = 0; j < examples[i].Doutput.size(); j++) { //salvo gli output
				file << examples[i].Doutput[j];
				if (j == examples[i].Doutput.size() - 1) { file << '\n'; }
				else { file << " "; }
			}
		}

	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////// FUNZIONI MODIFICA RETE//////////////////////////////////////////////////////////
	void deleteArc(int Nlayer, int Ncolumn, int targetLayer, int targetColumn) {
		for (int i = 0; i < Layers[Nlayer].Neurons[Ncolumn].numOutArcs; i++) {
			if (Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->layer == targetLayer && Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->column == targetColumn) {

				//modifico i parametri della rete relazionati a tale arco
				Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->numInArcs--;
				if (Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->influenceInput.size() > 0)Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->influenceInput.pop_back();
				if (Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->influenceOutput.size() > 0)Layers[Nlayer].Neurons[Ncolumn].influenceOutput.pop_back();

				if (Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->numInArcs == 0 && Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->layer != 0) {
					deleteNeuron(Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->layer, Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->column);
				}
				else {
					Layers[Nlayer].Neurons[Ncolumn].OutArcs.erase(Layers[Nlayer].Neurons[Ncolumn].OutArcs.begin() + i);
					Layers[Nlayer].Neurons[Ncolumn].numOutArcs--;
				}

				return;
			}
		}
		cout << "errore: arco (" << Nlayer << "-" << Ncolumn << ") -> (" << targetLayer << "-" << targetColumn << ") non trovato, eliinazione fallita" << endl;
	}
	void deleteArcByRef(int Nlayer, int Ncolumn, arc *targetArc) { //elimina un arco con gli indici della base e il puntatore alla connsessione

		//modifico i parametri della rete relazionati a tale arco
		targetArc->target->numInArcs--;
		if (targetArc->target->influenceInput.size() > 0)targetArc->target->influenceInput.pop_back();
		if (Layers[Nlayer].Neurons[Ncolumn].influenceOutput.size() > 0)Layers[Nlayer].Neurons[Ncolumn].influenceOutput.pop_back();

		if (targetArc->target->numInArcs == 0) {
			deleteNeuron(targetArc->target->layer, targetArc->target->column);
		}
		else {
			Layers[Nlayer].Neurons[Ncolumn].OutArcs.erase(Layers[Nlayer].Neurons[Ncolumn].OutArcs.begin() + getOutConTargetID(getTarget(Nlayer, Ncolumn), targetArc->target));
			Layers[Nlayer].Neurons[Ncolumn].numOutArcs--;
		}

	}
	void deleteNeuron(int Nlayer, int Ncolumn) {
		// elimino l'arco dai neuroni che puntano al nodo da eliminare e
		// decremento il numero di archi in output hai neuroni con come target il neurone da eliminare
		for (int i = 0; i < Nlayer; i++) {
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				for (int k = 0; k < Layers[i].Neurons[j].numOutArcs; k++) {
					if (Layers[i].Neurons[j].OutArcs[k].target->layer == Nlayer && Layers[i].Neurons[j].OutArcs[k].target->column == Ncolumn) {
						deleteArcByRef(i, j, &Layers[i].Neurons[j].OutArcs[k]);
					}
				}
			}
		}
		// eliminio gli archi del neurone da eliminare
		for (int i = 0; i < Layers[Nlayer].Neurons[Ncolumn].numOutArcs; i++) {
			deleteArcByRef(Nlayer, Ncolumn, &Layers[Nlayer].Neurons[Ncolumn].OutArcs[i]);
		}

		// decremento l'indice colonna dei neuroni successivi nello stesso layer
		for (int i = Ncolumn + 1; i < Layers[Nlayer].numNeurons; i++) Layers[Nlayer].Neurons[i].column--;

		// elimino il neurone
		Layers[Nlayer].Neurons.erase(Layers[Nlayer].Neurons.begin() + Ncolumn);
		Layers[Nlayer].numNeurons--;
		nNeurons--;
	}
	void addArc(int Nlayer, int Ncolumn, int targetLayer, int targetColumn) {
		arc newArc; newArc.target = &(Layers[targetLayer].Neurons[targetColumn]); newArc.weight = 0.01f;
		Layers[targetLayer].Neurons[targetColumn].numInArcs++;
		Layers[Nlayer].Neurons[Ncolumn].OutArcs.push_back(newArc);
		Layers[Nlayer].Neurons[Ncolumn].numOutArcs++;
	}
	void addNeuron(int Nlayer, float inConFill, float outConFill) {
		//init new neuron
		Neuron newNeur;
		newNeur.layer = Nlayer;
		newNeur.column = Layers[Nlayer].numNeurons; // column start at 0 , numNeurons start at 1
		int nBackNeurons = 0; for (int i = 0; i < Nlayer; i++) nBackNeurons += Layers[i].numNeurons;
		int nFrontNeurons = 0; for (int i = Nlayer + 1; i < nLayers; i++) nFrontNeurons += Layers[i].numNeurons;
		newNeur.numOutArcs = (int)(outConFill*nFrontNeurons);
		newNeur.numInArcs = (int)(inConFill*nBackNeurons);
		newNeur.OutArcs.resize(newNeur.numOutArcs);

		//bind neuron with neuron of front layers (OUTPUT CONNECTIONS)
		int x, y = 0;
		vector<u_int> OutArcAcces = casualVector(newNeur.numOutArcs);
		vector<u_int> NetAcces = casualVector(nFrontNeurons);
		while (y < OutArcAcces.size()) {
			x = 0;
			for (int i = Nlayer + 1; i < nLayers; i++) {
				for (int j = 0; j < Layers[i].numNeurons; j++) {
					if (y > OutArcAcces.size() - 1) continue;
					if (NetAcces[y] == x) {
						newNeur.OutArcs[OutArcAcces[y]].target = &(Layers[i].Neurons[j]);
						newNeur.OutArcs[OutArcAcces[y]].weight = 0.01f;
						y++;
					}
					x++;
				}
			}
		}
		cout << "connessioni output create" << endl << endl;
		WeightsStamp("w");

		// inserisco il neurone nella rete
		//NOTA push_back() reinizializa il vettore modificandi i puntatori alle celle, tutte le connessioni al layer che viene aggiornato vengono perse
		vector<conMap> connections = saveConsTowardsLyr(Nlayer); // salvo le connessioni al layer
		Layers[Nlayer].Neurons.push_back(newNeur); // immetto il nuovo neurone (i riferimenti del vettore cambiano)
		Layers[Nlayer].numNeurons++;
		loadConsTowardsLyr(connections); // aggiorno i puntatori collegati ai neuroni di questo vettore


		WeightsStamp("w");
		cout << "neurone inserito" << endl << endl;

		// bind neuron with back layers (INPUT CONNECTIONS)	
		NetAcces = casualVector(nBackNeurons);
		arc newArc; newArc.weight = 0.01f;
		y = 0;
		while (y < newNeur.numInArcs) {
			x = 0;
			for (int i = 0; i < Nlayer; i++) {
				for (int j = 0; j < Layers[i].numNeurons; j++) {
					if (y > newNeur.numInArcs - 1) continue;
					if (NetAcces[y] == x) {
						newArc.target = &(Layers[Nlayer].Neurons[Layers[Nlayer].numNeurons - 1]);
						//Layers[Nlayer].Neurons[Layers[Nlayer].numNeurons - 1].numInArcs++;
						Layers[i].Neurons[j].OutArcs.push_back(newArc);
						Layers[i].Neurons[j].numOutArcs++;
						cout << "BIND:" << y << " (" << i << "-" << j << ")  ->  (" << Nlayer << "-" << Layers[Nlayer].numNeurons - 1 << ") " << endl;
						WeightsStamp("w");
						y++;
					}
					x++;
				}
			}
		}
		nNeurons++;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////FUNZIONI VARIE///////////////////////////////////////////////////////////
	float DsigOut(int Layer, int Neuron) {
		//derivata della funzione sigmoide (Y*(1-Y)) 
		//RICORDA di aggiungere k se implementi la sensibilità della sigmoide
		return Layers[Layer].Neurons[Neuron].output*(1 - Layers[Layer].Neurons[Neuron].output);
	}
	float sigmoid(float x) { return 1 / (1 + exp(-x)); } // Sigmoide
	float logit(float x) { return log(x / (1 - x)); } // funzione sigmoide inversa 
	float gaussian(float x, float mu, float var) { return (1 / (var*sqrt(2 * M_PI)))*exp(-((pow(x - mu, 2.0)) / (2 * var))); } // Gaussiana(float x, float mu, float var)
	void WeightsStamp(string mode = "a") {
		//m - stampa le medie dei pesi di ogni layer
		//a - stampa tutti i pesi della rete con alcuni parametri di apprendimento
		//w - stampa tutti i pesi con il riferimento riga colonna al target
		//fc - stampa le medie dei gruppi di pesi tra due layer 
		if (mode == "m") { // medie dei layer
			float mean;
			int x;
			for (int i = 0; i < numLayers() - 1; i++) {
				x = 0;
				mean = 0;
				for (int c = 0; c < numNeurons(i); c++) {
					for (int d = 0; d < numCon(i, c); d++) {
						mean += getWeight(i, c, d);
						x++;
					}
				}
				mean /= x;
				cout << "Layer " << i << " weights:" << mean << endl;
			}
		}
		else if (mode == "a") { // stampa tutti i pesi e i parametri dei neuroni
			for (int i = 0; i < numLayers() - 1; i++) {
				for (int c = 0; c < numNeurons(i); c++) {
					cout << "(" << i << "-" << c << ")  output: " << getOutput(i, c) << "  BPerr: " << getBPerr(i, c) << endl;
					for (int d = 0; d < numCon(i, c); d++) {
						cout << getWeight(i, c, d) << " (" << getDeltaWeight(i, c, d) << ")  ";
					}
					cout << endl;
				}
			}
		}
		else if (mode == "w") { // stampa tutti i pesi con il riferimento al target
			for (int i = 0; i < numLayers() - 1; i++) {
				for (int c = 0; c < numNeurons(i); c++) {
					cout << "(" << i << "-" << c << ")" << "Con: " << numCon(i, c) << "  InCon: " << numInCon(i, c) << endl;
					for (int d = 0; d < numCon(i, c); d++) {
						cout << getWeight(i, c, d) << " (" << getConTargetLyr(i, c, d) << "," << getConTargetCol(i, c, d) << ")  ";
					}
					cout << endl;
				}
			}
		}
		else if (mode == "fc") { // stampa le medie dei pesi tra layer e layer
			vector<float> wLyr(nLayers - 1);
			vector<int> nwLyr(nLayers - 1);
			float wl = 0;
			for (int i = 0; i < nLayers - 1; i++) {
				for (int j = 0; j < Layers[i].numNeurons; j++) {
					for (int k = 0; k < Layers[i].Neurons[j].numOutArcs; k++) {
						wLyr[Layers[i].Neurons[j].OutArcs[k].target->layer - 1] += Layers[i].Neurons[j].OutArcs[k].weight;
						nwLyr[Layers[i].Neurons[j].OutArcs[k].target->layer - 1]++;
					}
				}
				cout << "Layer " << i << " weights: ";
				for (int j = 0; j < wLyr.size(); j++) {
					wLyr[j] /= nwLyr[j];
					if (j + 1 > i)cout << "(" << i << ", " << j + 1 << ") [" << nwLyr[j] << "] " << wLyr[j] << "  ";
				}
				cout << endl;
				fill(wLyr.begin(), wLyr.end(), 0);
				fill(nwLyr.begin(), nwLyr.end(), 0);
			}
		}
		else {
			cout << "WeightsStamp() argument error!" << endl;
		}
		cout << endl;
	}
	void sigLayer(int lyr) {  // applica la sigmoide a tutti i campi output dei neuroni nel layer specificato
		for (int i = 0; i < Layers[lyr].numNeurons; i++) {
			Layers[lyr].Neurons[i].output = sigmoid(Layers[lyr].Neurons[i].output);
		}
	}
	void bayesLayer(int lyr, bool absSum = false) {//applica il bayes all'output di ogni neurone del dato layer
		for (int i = 0; i < Layers[lyr].numNeurons; i++) {
			Layers[lyr].Neurons[i].output += Layers[lyr].Neurons[i].bayes;
		}
		// se la variabile absSum è true sommo il bayes in valore assoluto alla variabile absOutSum
		if (absSum == true) { for (int i = 0; i < Layers[lyr].numNeurons; i++) { Layers[lyr].Neurons[i].absOutSum += abs(Layers[lyr].Neurons[i].bayes); } }
	}
	void resetPotential() {
		// esegue il reset del potenziale di tutti i neuroni della rete
		for (int i = 0; i < nLayers; i++) {
			for (int j = 0; j < numNeurons(i); j++) {
				Layers[i].Neurons[j].output = 0;
			}
		}
	}
	void resetAbsSumPotenzial() {
		// esegue il reset della sommatoria di ogni input in valore assoluto di ogni neurone
		for (int i = 0; i < nLayers; i++) {
			for (int j = 0; j < numNeurons(i); j++) {
				Layers[i].Neurons[j].absOutSum = 0;
			}
		}
	}
	void resetAbsSumDelta() {
		// esegue il reset della sommatoria di ogni input in valore assoluto di ogni neurone
		for (int i = 0; i < nLayers; i++) {
			for (int j = 0; j < numNeurons(i); j++) {
				Layers[i].Neurons[j].absDeltaSum = 0;
			}
		}
	}
	void resetBPerr() { // esegue il reset dell'errore retropropagato in ogni neurone
		for (int i = 0; i < nLayers; i++) {
			for (int j = 0; j < numNeurons(i); j++) {
				Layers[i].Neurons[j].BPerr = 0;
			}
		}
	}
	void resetNeuronsID() {
		for (int i = 0; i < nLayers; i++) {
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				Layers[i].Neurons[j].layer = i;
				Layers[i].Neurons[j].column = j;
			}
		}
	}
	vector<conMap> saveConsTowardsLyr(int Layer) { // salva su vettore i riferimenti numerici delle connesioni verso il layer specificato
		vector<conMap> connections;
		conMap con;
		for (int i = 0; i < Layer; i++) {
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				for (int k = 0; k < Layers[i].Neurons[j].numOutArcs; k++) {
					if (Layers[i].Neurons[j].OutArcs[k].target->layer == Layer) {
						con.startCol = j;
						con.startLyr = i;
						con.arcRef = k;
						con.targetCol = Layers[i].Neurons[j].OutArcs[k].target->column;
						con.targetLyr = Layers[i].Neurons[j].OutArcs[k].target->layer;
						connections.push_back(con);
					}
				}
			}
		}
		return connections;
	}
	void loadConsTowardsLyr(vector<conMap> con) { // ricarica i riferimenti numerici delle connesioni verso il layer specificato
		for (int i = 0; i < con.size(); i++) {
			Layers[con[i].startLyr].Neurons[con[i].startCol].OutArcs[con[i].arcRef].target = getTarget(con[i].targetLyr, con[i].targetCol);
		}
	}
	void ClearDataset() { examples.clear(); }
	void genTestDataset(int nExe, int nIn, int nOut, float step, int type, float offset) { //generazione di una serie storica del seno (DEBUG)
		cout << "creating examples.." << endl;
		examples.resize(nExe);
		float x = 0, x2 = 0;
		for (int i = 0; i < nExe; i++) {
			examples[i].input.resize(nIn);
			examples[i].Doutput.resize(nOut);
			switch (type) {
			case 0: // debug dataset
				for (int j = nIn - 1; j >= 0; j--) {
					examples[i].input[j] = 1 + offset;
				}
				for (int j = 0; j < nOut; j++) {
					examples[i].Doutput[j] = 1 + offset;
				}
				break;
			case 1: // funzione predizione
				for (int j = 0; j < nIn; j++) {
					examples[i].input[j] = sin(x2) + offset;
					x2 += step;
				}
				for (int j = 0; j < nOut; j++) {
					examples[i].Doutput[j] = sin(x2) + offset;
					x2 += step;
				}
				x += step;
				x2 = x;
				break;
			case 2: // funzione uguale
				for (int j = 0; j < nIn; j++) {
					examples[i].input[j] = sin(x2) + offset;
					x2 += step;
				}
				x2 = x;
				for (int j = 0; j < nOut; j++) {
					examples[i].Doutput[j] = sin(x2) + offset;
					x2 += step;
				}
				x += step;
				x2 = x;
				break;
			case 3: // funzione specchio
				for (int j = nIn - 1; j >= 0; j--) {
					examples[i].input[j] = sin(x2) + offset;
					x2 += step;
				}
				x2 = x;
				for (int j = 0; j < nOut; j++) {
					examples[i].Doutput[j] = sin(x2) + offset;
					x2 += step;
				}
				x += step;
				x2 = x;
				break;
			}
		}
	}
	void setNetMap(float max, float min) {
		map.resize(Layers[nLayers - 1].numNeurons);
		for (int i = 0; i < map.size(); i++) {
			map[i].maxValue = max;
			map[i].minValue = min;
		}
	}
	float reverseMap(int neur, float val) {
		return (val - map[neur].minValue) / (map[neur].maxValue - map[neur].minValue);
	}
	vector<u_int> casualVector(int in, int start = 0) {
		// crea un vettore di n ellementi successivi e li disordina
		// creazione di una tabella di accesso casuale per un secondo vettore
		vector<u_int> out(in);
		for (int i = 0; i < out.size(); i++) out[i] = i + start;
		random_shuffle(out.begin(), out.end());
		return out;
	}
	/*vector<T, A>*/
	template<typename T, typename A>
	void shackeVector(vector<T, A> const& vec) {
		//esegue il mescolamento degli elementi all'interno di un oggetto vector 
		random_shuffle(vec.begin(), vec.end());
		//return vec;
	}
	void refreshNeurIdx() {
		int idx = 0;
		for (int i = 0; i < nLayers; i++) {
			for (int j = 0; j < numNeurons(i); j++) {
				getTarget(i, j)->neurIdx = idx++;
			}
		}
	}
	void datasetOffset(float offset) {
		for (int i = 0; i < examples.size(); i++) {
			for (int j = 0; j < examples[i].Doutput.size(); j++) {
				examples[i].Doutput[j] += offset;
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////ACCESSO A VARIABILI PRIVATE///////////////////////////////////////////////////
	int numLayers() { return nLayers; }
	int numNeurons(int Layer) { return Layers[Layer].numNeurons; }
	int numCon(int Layer, int Neuron) { return Layers[Layer].Neurons[Neuron].numOutArcs; }
	int numInCon(int Layer, int Neuron) { return Layers[Layer].Neurons[Neuron].numInArcs; }
	int getConTargetLyr(int Layer, int Neuron, int Arc) { return (int)Layers[Layer].Neurons[Neuron].OutArcs[Arc].target->layer; }
	int getConTargetCol(int Layer, int Neuron, int Arc) { return (int)Layers[Layer].Neurons[Neuron].OutArcs[Arc].target->column; }
	float getWeight(int Layer, int Neuron, int Arc) { return Layers[Layer].Neurons[Neuron].OutArcs[Arc].weight; }
	float getDeltaWeight(int Layer, int Neuron, int Arc) { return Layers[Layer].Neurons[Neuron].OutArcs[Arc].oldDelta; }
	float getOutput(int Layer, int Neuron) { return Layers[Layer].Neurons[Neuron].output; }
	float getBPerr(int Layer, int Neuron) { return Layers[Layer].Neurons[Neuron].BPerr; }
	ptNeuron getTarget(int Layer, int Neuron) { return &(Layers[Layer].Neurons[Neuron]); }
	ptNeuron getConTarget(int Layer, int Neuron, int Conn) { return Layers[Layer].Neurons[Neuron].OutArcs[Conn].target; }
	int getOutConTargetID(ptNeuron base, ptNeuron target) {
		for (int i = 0; i < base->numOutArcs; i++) {
			if (base->OutArcs[i].target == target) {
				return i;
			}
		}
		cout << "connessione non trovata ERRORE!" << endl;
		return -1;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////WINDOWS HIGH SPEED TIMING//////////////////////////////////////////////////////
	BOOL WINAPI QueryPerformanceCounter(_Out_ LARGE_INTEGER *lpPerformanceCount);
	BOOL WINAPI QueryPerformanceFrequency(_Out_ LARGE_INTEGER *lpFrequency);
	inline long long PerformanceCounter() noexcept
	{
		LARGE_INTEGER li;
		::QueryPerformanceCounter(&li);
		return li.QuadPart;
	}
	inline long long PerformanceFrequency() noexcept
	{
		LARGE_INTEGER li;
		::QueryPerformanceFrequency(&li);
		return li.QuadPart;
	}
	/* HOW TO USE:
	long long t0 = PerformanceCounter();
	//code to bench..
	long long t1 = PerformanceCounter();
	double elapsedMilliseconds = ((t1 - t0) * 1000.0) / PerformanceFrequency();
	*/
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

class MLP : public Network {
protected:
	//Hopfield *Supporter;

public:

	float NetPerformance = 0; // tempo di esecuzione medio in millisecondi
	float NetErrPercent = 0; //errore percentuale medio associato alla rete

	MLP(string filename) :Network(filename) {};

	////////////////////////////////////////////////FUNZIONI CREAZIONE RETE///////////////////////////////////////////////////////
	//CREAZIONE RETE QUADRATA
	void qubeNet(int Nlayers, int Ncolumns, int input, int output, bool c, float initValue = 0.01f) { //TODO inserire n° neuroni di input e output!!
		// testata e funzionante
		// nlayers - numero layer della rete
		// Ncolumns - numero neuroni per strato
		// c - se true inizializa casualmente i pesi altrimenti sono inizializati tutti a 1
		Layers.resize(Nlayers, Layer()); //= new vector<Layer>(stoi(segment)); // dichiarazione numero di layer
		nLayers = Nlayers;
		cout << "Declaring structure...." << endl;

		//per lo strato di input
		Layers[0].numNeurons = input;
		Layers[0].Neurons.resize(input, Neuron());
		nNeurons += input;
		for (int j = 0; j < input; j++) {
			Layers[0].Neurons[j].OutArcs.resize(Ncolumns, arc());
			Layers[0].Neurons[j].numOutArcs = Ncolumns;
			Layers[0].Neurons[j].layer = 0;
			Layers[0].Neurons[j].column = j;
			nArc += Ncolumns;
		}

		for (int i = 1; i < Nlayers - 1; i++) {
			// per gli strati intermedi
			Layers[i].numNeurons = Ncolumns;
			Layers[i].Neurons.resize(Ncolumns, Neuron());
			nNeurons += Ncolumns;
			for (int j = 0; j < Ncolumns; j++) {
				if (i < Nlayers - 2) {
					Layers[i].Neurons[j].OutArcs.resize(Ncolumns, arc());
					Layers[i].Neurons[j].numOutArcs = Ncolumns;
					nArc += Ncolumns;
				}
				else {
					Layers[i].Neurons[j].OutArcs.resize(output, arc());
					Layers[i].Neurons[j].numOutArcs = output;
					nArc += output;
				}
				Layers[i].Neurons[j].layer = i;
				Layers[i].Neurons[j].column = j;
			}

		}
		//per lo strato di output
		Layers[Nlayers - 1].numNeurons = output;
		Layers[Nlayers - 1].Neurons.resize(output, Neuron());
		nNeurons += output;
		for (int j = 0; j < output; j++) {
			Layers[Nlayers - 1].Neurons[j].layer = Nlayers - 1;
			Layers[Nlayers - 1].Neurons[j].column = j;
		}

		float initWeight = initValue;
		float Q = sqrt(3)*sqrt(2);
		int M = 10;
		cout << "initializing wheights...." << endl;

		for (int j = 0; j < input; j++) {
			for (int k = 0; k < Ncolumns; k++) {
				if (c) { initWeight = 0; while (abs(initWeight) > 0.05) { srand(clock() - 80); initWeight = (rand() % M) / Q; } }
				Layers[0].Neurons[j].OutArcs[k].target = &(Layers[1].Neurons[k]);
				Layers[0].Neurons[j].OutArcs[k].weight = initWeight;
				Layers[1].Neurons[k].numInArcs++;
			}
		}

		for (int i = 1; i < Nlayers - 2; i++) {
			//cout << "Layer:" << i << endl;
			for (int j = 0; j < Ncolumns; j++) {
				for (int k = 0; k < Ncolumns; k++) {
					if (c) { initWeight = 0; while (abs(initWeight) > 0.05) { srand(clock() + 50); initWeight = (rand() % M) / Q; } }
					Layers[i].Neurons[j].OutArcs[k].target = &(Layers[i + 1].Neurons[k]);
					Layers[i].Neurons[j].OutArcs[k].weight = initWeight;
					Layers[i + 1].Neurons[k].numInArcs++;
				}
			}
		}

		for (int j = 0; j < Ncolumns; j++) {
			for (int k = 0; k < output; k++) {
				if (c) { initWeight = 0; while (abs(initWeight) > 0.05) { srand(clock() / 0.7556); initWeight = (rand() % M) / Q; } }
				Layers[Nlayers - 2].Neurons[j].OutArcs[k].target = &(Layers[Nlayers - 1].Neurons[k]);
				Layers[Nlayers - 2].Neurons[j].OutArcs[k].weight = initWeight;
				Layers[Nlayers - 1].Neurons[k].numInArcs++;
			}
		}
		refreshNeurIdx();
	}
	//CREAZIONE RETE QUADRATA COMPLETAMENTE CONNESSA
	void qubeNetFC(int Nlayers, int Ncolumns, int input, int output, bool c, float initValue = 0.01f) {
		//TODO inserire n° neuroni di input e output!!
		// testata e funzionante
		// nlayers - numero layer della rete
		// Ncolumns - numero neuroni per strato
		// c - se true inizializa casualmente i pesi altrimenti sono inizializati tutti a 1
		Layers.resize(Nlayers, Layer()); //= new vector<Layer>(stoi(segment)); // dichiarazione numero di layer
		nLayers = Nlayers;
		cout << "Declaring structure...." << endl;

		//per lo strato di input
		Layers[0].numNeurons = input;
		Layers[0].Neurons.resize(input, Neuron());
		nNeurons += input;
		int numA = (nLayers - 2)*Ncolumns + output;
		nArc += input * numA;
		for (int j = 0; j < input; j++) {
			Layers[0].Neurons[j].OutArcs.resize(numA, arc());
			Layers[0].Neurons[j].numOutArcs = numA;
			Layers[0].Neurons[j].layer = 0;
			Layers[0].Neurons[j].column = j;
		}

		for (int i = 1; i < Nlayers - 1; i++) {
			// per gli strati intermedi
			numA = (Nlayers - i - 2)*Ncolumns + output;
			Layers[i].numNeurons = Ncolumns;
			Layers[i].Neurons.resize(Ncolumns, Neuron());
			nNeurons += Ncolumns;
			for (int j = 0; j < Ncolumns; j++) {
				if (i < Nlayers - 2) {
					Layers[i].Neurons[j].OutArcs.resize(numA, arc());
					Layers[i].Neurons[j].numOutArcs = numA;
					nArc += numA;
				}
				else {
					Layers[i].Neurons[j].OutArcs.resize(output, arc());
					Layers[i].Neurons[j].numOutArcs = output;
					nArc += output;
				}
				Layers[i].Neurons[j].layer = i;
				Layers[i].Neurons[j].column = j;
			}

		}
		//per lo strato di output
		Layers[Nlayers - 1].numNeurons = output;
		Layers[Nlayers - 1].Neurons.resize(output, Neuron());
		nNeurons += output;
		for (int j = 0; j < output; j++) {
			Layers[Nlayers - 1].Neurons[j].layer = Nlayers - 1;
			Layers[Nlayers - 1].Neurons[j].column = j;
		}

		float initWeight = initValue;
		float Q = sqrt(3)*sqrt(2);
		int M = 10;
		int arcRef = 0;
		cout << "initializing weights...." << endl;

		// inizializzo pesi strato input
		for (int j = 0; j < input; j++) {
			for (int k = 0; k < (Nlayers - 2); k++) {
				for (int n = 0; n < Ncolumns; n++) {
					numA = k * Ncolumns + n;
					if (c) { initWeight = 0; while (abs(initWeight) > 0.05) { srand(clock() - 80); initWeight = (rand() % M) / Q; } }
					Layers[0].Neurons[j].OutArcs[numA].target = &(Layers[k + 1].Neurons[n]);
					Layers[0].Neurons[j].OutArcs[numA].weight = initWeight;
					Layers[k + 1].Neurons[n].numInArcs++;
				}
			}
			for (int n = 0; n < output; n++) {
				numA = (Nlayers - 2) * Ncolumns + n;
				if (c) { initWeight = 0; while (abs(initWeight) > 0.05) { srand(clock() - 80); initWeight = (rand() % M) / Q; } }
				Layers[0].Neurons[j].OutArcs[numA].target = &(Layers[Nlayers - 1].Neurons[n]);
				Layers[0].Neurons[j].OutArcs[numA].weight = initWeight;
				Layers[Nlayers - 1].Neurons[n].numInArcs++;
			}
		}
		//inizializzo pesi strati intermedi
		for (int i = 1; i < Nlayers - 2; i++) {
			//cout << "Layer:" << i << endl;
			for (int j = 0; j < Ncolumns; j++) {
				for (int k = i + 1; k < Nlayers - 1; k++) {
					for (int w = 0; w < Ncolumns; w++) {
						numA = (k - i - 1)*Ncolumns + w;
						if (c) { initWeight = 0; while (abs(initWeight) > 0.05) { srand(clock() + 50); initWeight = (rand() % M) / Q; } }
						Layers[i].Neurons[j].OutArcs[numA].target = &(Layers[k].Neurons[w]);
						Layers[i].Neurons[j].OutArcs[numA].weight = initWeight;
						Layers[k].Neurons[w].numInArcs++;
					}
				}
				for (int k = 0; k < output; k++) {
					numA = (Nlayers - 2 - i)*Ncolumns + k;
					if (c) { initWeight = 0; while (abs(initWeight) > 0.05) { srand(clock() + 50); initWeight = (rand() % M) / Q; } }
					Layers[i].Neurons[j].OutArcs[numA].target = &(Layers[Nlayers - 1].Neurons[k]);
					Layers[i].Neurons[j].OutArcs[numA].weight = initWeight;
					Layers[Nlayers - 1].Neurons[k].numInArcs++;
				}
			}
		}
		//inizializzo pesi strato output
		for (int j = 0; j < Ncolumns; j++) {
			for (int k = 0; k < output; k++) {
				if (c) { initWeight = 0; while (abs(initWeight) > 0.05) { srand(clock() / 0.7556); initWeight = (rand() % M) / Q; } }
				Layers[Nlayers - 2].Neurons[j].OutArcs[k].target = &(Layers[Nlayers - 1].Neurons[k]);
				Layers[Nlayers - 2].Neurons[j].OutArcs[k].weight = initWeight;
				Layers[Nlayers - 1].Neurons[k].numInArcs++;
			}
		}
		refreshNeurIdx();
	}
	//CREAZIONE RETE CUSTOM
	void customNet(int Nlayers, vector<int> Ncolumns, float conFill) {
		//TODO aggiungere la conta delle connessioni in nArc durante la dichiarazione
		if (Ncolumns.size() != Nlayers) { cout << "costumNet() FAILED! parameters error" << endl; return; } //errore
																											//dichiarazione layers
		Layers.resize(Nlayers);
		nLayers = Nlayers;
		//dichiarazione neuroni per layer
		for (int i = 0; i < Nlayers; i++) {
			Layers[i].Neurons.resize(Ncolumns[i]);
			Layers[i].numNeurons = Ncolumns[i];
			nNeurons += Ncolumns[i];
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				Layers[i].Neurons[j].layer = i;
				Layers[i].Neurons[j].column = j;
			}
		}
		//dichiarazione connessioni in uscita di ogni neurone
		for (int i = 0; i < nLayers - 1; i++) {
			cout << "Layer " << i << endl;
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				cout << "Neuron " << j << endl;
				int nFrontNeurons = 0; for (int g = i + 1; g < nLayers; g++) nFrontNeurons += Layers[g].numNeurons;
				Layers[i].Neurons[j].numOutArcs = conFill * nFrontNeurons;
				Layers[i].Neurons[j].OutArcs.resize(conFill* nFrontNeurons);

				vector<u_int> OutArcAcces = casualVector(Layers[i].Neurons[j].numOutArcs);
				vector<u_int> NetAcces = casualVector(nFrontNeurons);
				int x, y = 0;

				while (y < OutArcAcces.size()) {
					x = 0;
					for (int k = i + 1; k < nLayers; k++) {
						for (int w = 0; w < Layers[k].numNeurons; w++) {
							if (y > OutArcAcces.size() - 1) continue;
							if (NetAcces[y] == x) {
								Layers[i].Neurons[j].OutArcs[OutArcAcces[y]].target = &(Layers[k].Neurons[w]);
								Layers[i].Neurons[j].OutArcs[OutArcAcces[y]].weight = 0.01f;
								Layers[k].Neurons[w].numInArcs++;
								y++; cout << y << endl;
							}
							x++;
						}
					}
				}
			}
		}
		refreshNeurIdx();
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////STIMOLAZIONE RETE/////////////////////////////////////////////////////////////////
	//procedura di propagazione dell'informazione
	void inputNet(vector<float> &input, vector<float> &output) {
		if (input.size() != Layers[0].numNeurons) { cout << "il vettore in input deve avere dim " << Layers[0].numNeurons << endl; return; }
		if (output.size() != Layers[nLayers - 1].numNeurons) { cout << "il vettore in output deve avere dim " << Layers[nLayers - 1].numNeurons << endl; return; }

		// carico il vettore di input nella rete
		for (int i = 0; i < Layers[0].numNeurons; i++) {
			Layers[0].Neurons[i].output = input[i];
		}
		//sigLayer(0);

		// propagazione dello stimolo
		for (int i = 0; i < nLayers - 1; i++) {
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				for (int k = 0; k < Layers[i].Neurons[j].numOutArcs; k++) {
					Layers[i].Neurons[j].OutArcs[k].target->output += Layers[i].Neurons[j].output*Layers[i].Neurons[j].OutArcs[k].weight;
				}
			}
			bayesLayer(i + 1);
			sigLayer(i + 1);
		}

		float delta;

		if (map.size() == 0) { cout << "Errore ... la mappatura degli output non è stata settata!!"; return; }

		for (int i = 0; i < Layers[nLayers - 1].numNeurons; i++) {
			delta = map[i].maxValue - map[i].minValue;
			output[i] = ((Layers[nLayers - 1].Neurons[i].output)*delta) + map[i].minValue;
			Layers[nLayers - 1].Neurons[i].output = output[i];
		}
	}
	//esegue una propagazione dell'informazione salvando lo storico di propagazione degli input
	void inputNetProfiler(vector<float> &input, vector<float> &output) {
		if (input.size() != Layers[0].numNeurons) { cout << "il vettore in input deve avere dim " << Layers[0].numNeurons << endl; return; }
		if (output.size() != Layers[nLayers - 1].numNeurons) { cout << "il vettore in output deve avere dim " << Layers[nLayers - 1].numNeurons << endl; return; }

		// carico il vettore di input nella rete
		for (int i = 0; i < Layers[0].numNeurons; i++) {
			Layers[0].Neurons[i].output = input[i];
			//Layers[0].Neurons[i].absOutSum = abs(input[i]);
		}
		//sigLayer(0);
		// propagazione dello stimolo
		for (int i = 0; i < nLayers - 1; i++) {
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				for (int k = 0; k < Layers[i].Neurons[j].numOutArcs; k++) {
					Layers[i].Neurons[j].OutArcs[k].target->output += Layers[i].Neurons[j].output*Layers[i].Neurons[j].OutArcs[k].weight;
					Layers[i].Neurons[j].OutArcs[k].target->absOutSum += abs(Layers[i].Neurons[j].output*Layers[i].Neurons[j].OutArcs[k].weight);
				}
			}

			bayesLayer(i + 1, true);
			sigLayer(i + 1);
		}

		resetVectorsProfiler(true, false);// resetto tutti i vettori
		//inizializzo la propagazione dal primo layer
		int IDinf = 0;
		for (int i = 0; i < Layers[1].numNeurons; i++) { // scorro i neuroni del primo strato nascosto
			for (int j = 0; j < Layers[0].numNeurons; j++) { // scorro i neuroni dello strato di input 
				for (int k = 0; k < numCon(0, j); k++) { // scorro le connessioni del neurone dello strato di input
					if (&Layers[1].Neurons[i] == Layers[0].Neurons[j].OutArcs[k].target) {
						Layers[1].Neurons[i].influenceInput[IDinf] = (abs(getTarget(0, j)->output*getTarget(0, j)->OutArcs[k].weight) / (getTarget(0, j)->OutArcs[k].target->absOutSum));
						IDinf++;
					}
				}
			}
			IDinf = 0;
		}
		//inizzializzo la propagazione negli altri layer
		for (int t = 1; t < nLayers - 1; t++) { // togliere il meno uno
			IDinf = 0;
			for (int i = 0; i < Layers[t + 1].numNeurons; i++) {
				for (int j = 0; j < t + 1; j++) {
					for (int w = 0; w < Layers[j].numNeurons; w++) {
						for (int k = 0; k < numCon(j, w); k++) {
							if (&Layers[t + 1].Neurons[i] == Layers[j].Neurons[w].OutArcs[k].target) {
								Layers[t + 1].Neurons[i].influenceInput[IDinf] = (abs(getTarget(j, w)->output*getTarget(j, w)->OutArcs[k].weight) / (getTarget(j, w)->OutArcs[k].target->absOutSum));
								IDinf++;
							}
						}
					}
				}
				IDinf = 0;
			}
		}


		float delta;

		if (map.size() == 0) {
			cout << "Errore ... la mappatura degli output non è stata settata!!" << endl;
			return;
		}

		for (int i = 0; i < Layers[nLayers - 1].numNeurons; i++) {
			delta = map[i].maxValue - map[i].minValue;
			output[i] = ((Layers[nLayers - 1].Neurons[i].output)*delta) + map[i].minValue;
			Layers[nLayers - 1].Neurons[i].output = output[i];
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////FUNZIONI DI ADDESTRAMENTO MLP////////////////////////////////////////////////////////
	//Algoritmo di addestramento Back-propagation
	void BP(int iter, float eps, float beta, float ErrPercent) {
		float err = 0; // errore del singolo neurone
		float Err = 0; // errore quadratio del singolo esempio
		float Gerr = 0; // errore quadratico dell'intero dataset
		float Perr = 0;
		float delta = 0;
		long long t0 = 0;
		long long t1 = 0;
		long long inT1 = 0;
		double elapsedMilliseconds = 0;
		double executionTime = 0;
		double inputTime = 0;
		int x = 0;
		vector<float> Out(numNeurons(nLayers - 1)); // vettore supporto output rete
		vector<float> GerrStory;

		for (int t = 0; t < iter; t++) {
			Gerr = 0;
			Perr = 0;
			for (int e = 0; e < examples.size(); e++) {
				t0 = PerformanceCounter();
				// eseguo l'esempio
				inputNet(examples[e].input, Out);

				t1 = PerformanceCounter();
				elapsedMilliseconds = ((t1 - t0) * 1000.0) / PerformanceFrequency();
				inputTime += elapsedMilliseconds;

				// clacolo il delta delle variazioni dei pesi dello strato di output
				Err = 0;
				for (int r = 0; r < Out.size(); r++) {
					delta = map[r].maxValue - map[r].minValue;
					err = Out[r] - examples[e].Doutput[r];
					Perr += abs(err / examples[e].Doutput[r]) * 100; // errore percentuale
					Err += pow(err, 2);
					if (x == e && t > -1) {
						cout << "es." << x << " Y" << r << " = " << Out[r] << "   " << "D" << r << " = " << examples[e].Doutput[r] << "  err: " << abs(Out[r] - examples[e].Doutput[r]) << endl;
						if (r == Out.size() - 1) { x--; WeightsStamp("fc"); /*Sleep(5000);*/ }
						if (x < 0) x = examples.size() - 1;
					}
					//mErr = ((err- map[r].minValue)/ delta) ;
					// calcolo il delta della variazione dei pesi
					Layers[nLayers - 1].Neurons[r].BPerr = reverseMap(r, err) * (reverseMap(r, Out[r]) *(1 - reverseMap(r, Out[r])));
					//applico le correzioni ai bayes
					Layers[nLayers - 1].Neurons[r].oldBayesDelta = (-eps * (Layers[nLayers - 1].Neurons[r].BPerr)) + beta * Layers[nLayers - 1].Neurons[r].oldBayesDelta;
					Layers[nLayers - 1].Neurons[r].bayes += Layers[nLayers - 1].Neurons[r].oldBayesDelta;
				}
				//applico le correzioni ai pesi dello strato di output
				for (int i = 0; i < numNeurons(nLayers - 2); i++) {
					for (int j = 0; j < numCon(nLayers - 2, i); j++) {

						Layers[nLayers - 2].Neurons[i].OutArcs[j].oldDelta = (-eps * (Layers[nLayers - 2].Neurons[i].OutArcs[j].target->BPerr)*Layers[nLayers - 2].Neurons[i].output) + beta * Layers[nLayers - 2].Neurons[i].OutArcs[j].oldDelta;
						Layers[nLayers - 2].Neurons[i].OutArcs[j].weight += Layers[nLayers - 2].Neurons[i].OutArcs[j].oldDelta;
					}
				}
				// rieseguo la procedura per tutti gli altri strati
				for (int i = nLayers - 2; i > 0; i--) { // dal penultimo strato al secondo
														// clacolo il delta delle variazioni dei pesi dello strato i-1
					for (int j = 0; j < numNeurons(i); j++) {
						err = 0;
						for (int k = 0; k < numCon(i, j); k++) err += Layers[i].Neurons[j].OutArcs[k].target->BPerr * Layers[i].Neurons[j].OutArcs[k].weight;
						Layers[i].Neurons[j].BPerr = DsigOut(i, j)*err;
						//applico le correzioni ai bayes
						Layers[i].Neurons[j].oldBayesDelta = (-eps * (Layers[i].Neurons[j].BPerr)) + beta * Layers[i].Neurons[j].oldBayesDelta;
						Layers[i].Neurons[j].bayes += Layers[i].Neurons[j].oldBayesDelta;

					}
					// applico le correzioni ai pesi dello strato i-1
					for (int j = 0; j < numNeurons(i - 1); j++) {
						for (int k = 0; k < numCon(i - 1, j); k++) {
							Layers[i - 1].Neurons[j].OutArcs[k].oldDelta = (-eps * (Layers[i - 1].Neurons[j].OutArcs[k].target->BPerr)*Layers[i - 1].Neurons[j].output) + beta * Layers[i - 1].Neurons[j].OutArcs[k].oldDelta;
							Layers[i - 1].Neurons[j].OutArcs[k].weight += Layers[i - 1].Neurons[j].OutArcs[k].oldDelta;
						}
					}
				}
				Gerr += Err;
				resetPotential(); // azzero i potenziali di output della rete
				t1 = PerformanceCounter();
				elapsedMilliseconds = ((t1 - t0) * 1000.0) / PerformanceFrequency();
				executionTime += elapsedMilliseconds;
			}

			Gerr /= 2;
			Perr /= examples.size()*Out.size();
			executionTime /= examples.size();
			inputTime /= examples.size();
			NetPerformance = inputTime;
			NetErrPercent = Perr;
			cout << "Iterazione  " << t << " errore quadratico: " << Gerr << "   errore percentuale medio:  " << Perr << "  %err  exemple time: " << executionTime << " ms  inputTime: " << inputTime << " ms" << endl;

			//if (t > 100 && t < 200)Sleep(500);
			if (Perr < ErrPercent) {
				cout << "Percentuale di errore obbiettivo raggiunta!" << endl;
				return;
			}

		}
	}
	//esecuzione del backpropagation per un solo esempio
	void oneBP(float eps, float beta, example e) {
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////ALTRE FUNZIONI MLP///////////////////////////////////////////////////////////

	void initVectorsProfiler() {
		//inizializza i vettori di benchmark presenti nei neuroni
		//inizializzazione vettori del primo strato
		for (int j = 0; j < Layers[0].numNeurons; j++) {
			Layers[0].Neurons[j].influenceOutput.resize(Layers[0].Neurons[j].numOutArcs);
			fill(Layers[0].Neurons[j].influenceOutput.begin(), Layers[0].Neurons[j].influenceOutput.end(), 0);
		}
		for (int i = 1; i < nLayers - 1; i++) {
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				Layers[i].Neurons[j].influenceInput.resize(Layers[i].Neurons[j].numInArcs);
				Layers[i].Neurons[j].influenceOutput.resize(Layers[i].Neurons[j].numOutArcs);
				// resetto tutti i vettori  zero
				fill(Layers[i].Neurons[j].influenceInput.begin(), Layers[i].Neurons[j].influenceInput.end(), 0);
				fill(Layers[i].Neurons[j].influenceOutput.begin(), Layers[i].Neurons[j].influenceOutput.end(), 0);
			}
		}
		//inizializzazione dei vettori dell'ultimo strato
		for (int j = 0; j < Layers[nLayers - 1].numNeurons; j++) {
			Layers[nLayers - 1].Neurons[j].influenceInput.resize(Layers[nLayers - 1].Neurons[j].numInArcs);
			fill(Layers[nLayers - 1].Neurons[j].influenceInput.begin(), Layers[nLayers - 1].Neurons[j].influenceInput.end(), 0);
		}
	}
	void resetVectorsProfiler(bool inInfl, bool outInfl) { //resetta a zero tutti i vettori di profilazione esclusi layer input e output
		for (int i = 0; i < nLayers; i++) {
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				if (i > 0 && inInfl == true)fill(Layers[i].Neurons[j].influenceInput.begin(), Layers[i].Neurons[j].influenceInput.end(), 0);
				if (i < nLayers - 1 && outInfl == true)fill(Layers[i].Neurons[j].influenceOutput.begin(), Layers[i].Neurons[j].influenceOutput.end(), 0);
			}
		}
	}
	//stampa a schermo l'influenza degli input per ogni uscita
	void stampInputInfluences(bool all = false) {
		cout << endl;
		if (all == false) {
			for (int i = 0; i < Layers[nLayers - 1].numNeurons; i++) {
				cout << "Output (" << i << "): ";
				for (int j = 0; j < Layers[nLayers - 1].Neurons[i].influenceInput.size(); j++) {
					cout << "infl n in(" << j << "): " << Layers[nLayers - 1].Neurons[i].influenceInput[j] * 100 << "% || ";
				}
				cout << endl;
			}
		}
		else {
			int l = 0;
			int c = 0;
			for (int i = 0; i < nLayers; i++) {
				for (int j = 0; j < Layers[i].numNeurons; j++) {
					cout << "neuron (" << i << ", " << j << ") input influence: ";
					for (int k = 0; k < Layers[i].Neurons[j].influenceInput.size(); k++) {
						l = basePosfromtarget(getTarget(i, j), k)->layer;
						c = basePosfromtarget(getTarget(i, j), k)->column;
						cout << " (" << l << ", " << c << "): " << Layers[i].Neurons[j].influenceInput[k] * 100 << "% || ";// non mostra la correlazione con il neurone ma solo in numero di arco posizionale per il neurone
					}
					cout << endl;
				}

			}
		}
	}
	//stampa a schermo l'influenza degli errori degli output per ogni ingresso
	void stampOutputErrorPropagation(bool all = false) {
		cout << endl;
		if (all == false) {
			for (int i = 0; i < Layers[0].numNeurons; i++) {
				cout << "input (" << i << ") BPerrore : ";
				for (int j = 0; j < Layers[0].Neurons[i].influenceOutput.size(); j++) {
					cout << " (" << j << "): " << Layers[0].Neurons[i].influenceOutput[j] * 100 << "% || ";
				}
				cout << endl;
			}
		}
		else {
			for (int i = 0; i < nLayers; i++) {
				for (int j = 0; j < Layers[i].numNeurons; j++) {
					cout << "neuron (" << i << ", " << j << "): ";
					for (int k = 0; k < Layers[i].Neurons[j].influenceOutput.size(); k++) {
						cout << "BPerr n out(" << k << "): " << Layers[i].Neurons[j].influenceOutput[k] * 100 << "% || "; // non mostra la correlazione con il neurone ma solo in numero di arco posizionale per il neurone
					}
					cout << endl;
				}

			}
		}
	}
	// dal riferimento al nodo target e dall'indice del vettore di influenza restituisce il puntatore al neurone base
	ptNeuron basePosfromtarget(ptNeuron target, int k) {
		int k2 = -1;
		for (int i = 0; i < target->layer; i++) {
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				for (int w = 0; w < numCon(i, j); w++) {
					if (target == getConTarget(i, j, w)) {
						k2++;
						if (k2 == k) return getTarget(i, j);
					}
				}
			}
		}
		return nullptr;
	}
	//dato un neurone e l'indice di un suo arco restituisce l'indice di quella connessione all'interno del vettore influenceInput all'interno del neurone target
	int idBaseConReftoTargetInfl(ptNeuron base, int arc) {
		ptNeuron target = base->OutArcs[arc].target;
		int k2 = -1;
		for (int i = 0; i < target->layer; i++) {
			for (int j = 0; j < numNeurons(i); j++) {
				for (int k = 0; k < numCon(i, j); k++) {
					if (target == getConTarget(i, j, k)) {
						k2++;
						if (base == getTarget(i, j)) {
							return k2;
						}
					}
				}
			}
		}
		return -1;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

};

class Hopfield : public Network {

public:

	MLP* mlp; // puntatore alla rete da supportare
	vector<interArc> binds; // connessioni tra rete mlp e Hopfiled

	Hopfield(string filename, MLP *target = NULL) :Network(filename) {
		if (target != NULL) {
			mlp = target;
			supportNet();
		}
	}

	////////////////////////////////////////////////FUNZIONI CREAZIONE RETE Hopfield///////////////////////////////////////////////
	//Rete ad anello //RICORDA i pesi sono settati automaticamente a 1 TODO implementa controllo da parametro
	void toroidNet(int Nlayers, vector<int> Ncolumns, float conFill) {
		//sviluppo simile alle reti quadrate FC ma con connessioni anche tra i neuroni di uno stesso strato
		if (Ncolumns.size() != Nlayers) { cout << "ringNet() FAILED! parameters error" << endl; return; } //errore
																										  //dichiarazione layers
		Layers.resize(Nlayers);
		nLayers = Nlayers;
		//dichiarazione neuroni per layer
		for (int i = 0; i < Nlayers; i++) {
			Layers[i].Neurons.resize(Ncolumns[i]);
			Layers[i].numNeurons = Ncolumns[i];
			nNeurons += Ncolumns[i];
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				Layers[i].Neurons[j].layer = i;
				Layers[i].Neurons[j].column = j;
			}
		}
		//dichiarazione connessioni in uscita di ogni neurone
		for (int i = 0; i < nLayers; i++) {
			//cout << "Layer " << i << endl;
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				//cout << "Neuron " << j << endl;
				int nFrontNeurons = 0; for (int g = i; g < nLayers; g++) nFrontNeurons += Layers[g].numNeurons;
				Layers[i].Neurons[j].numOutArcs = conFill * nFrontNeurons;
				Layers[i].Neurons[j].OutArcs.resize(conFill* nFrontNeurons);

				vector<u_int> OutArcAcces = casualVector(Layers[i].Neurons[j].numOutArcs);
				vector<u_int> NetAcces = casualVector(nFrontNeurons);

				int x, y = 0;

				while (y < OutArcAcces.size()) {
					x = 0;
					for (int k = i; k < nLayers; k++) {
						for (int w = 0; w < Layers[k].numNeurons; w++) {
							if (y > OutArcAcces.size() - 1) continue;
							if (NetAcces[y] == x) {
								Layers[i].Neurons[j].OutArcs[OutArcAcces[y]].target = &(Layers[k].Neurons[w]);
								Layers[i].Neurons[j].OutArcs[OutArcAcces[y]].weight = 1.0f;
								Layers[k].Neurons[w].numInArcs++;
								y++; cout << y << endl;
							}
							x++;
						}
					}
				}
				//TOPPA elimino le connessioni dei neuroni con se stessi
				for (int k = 0; k < Layers[i].Neurons[j].numOutArcs; k++) {
					if (Layers[i].Neurons[j].OutArcs[k].target->layer == i) {
						if (Layers[i].Neurons[j].OutArcs[k].target->column == j) {
							deleteArc(i, j, i, j);
						}
					}
				}
			}
		}
	}
	//Genera una rete di supporto all'apprendimento su misura per una mlp 
	void supportNet() {
		//creo la rete di supporto
		nLayers = 1;
		nNeurons = mlp->nNeurons;
		nArc = mlp->nArc;
		Layers.resize(1);
		Layers[0].Neurons.resize(nNeurons);
		Layers[0].numNeurons = nNeurons;
		//collego la rete di supporto alla mlp
		//e copio le connessioni dalla mlp sulla hopfield
		binds.resize(mlp->nNeurons);
		int bindPos = 0;
		for (int i = 0; i < mlp->nLayers; i++) {
			for (int j = 0; j < mlp->numNeurons(i); j++) {
				binds[bindPos].base = getTarget(0, bindPos);
				binds[bindPos].base->layer = 0;
				binds[bindPos].base->column = bindPos;
				binds[bindPos].target = mlp->getTarget(i, j);
				binds[bindPos].base->numOutArcs = binds[bindPos].target->numOutArcs;
				binds[bindPos].base->numInArcs = binds[bindPos].target->numInArcs;
				binds[bindPos].base->OutArcs.resize(binds[bindPos].target->numOutArcs);
				bindPos++;
			}
		}

		for (int i = 0; i < binds.size(); i++) {
			for (int k = 0; k < binds[i].target->numOutArcs; k++) {
				int TargetPos = searchFromTarget(binds[i].target->OutArcs[k].target);
				if (TargetPos != -1) {
					binds[i].base->OutArcs[k].target = binds[TargetPos].base;
					binds[i].base->OutArcs[k].weight = 1.0f;
				}
			}
		}
	}
	//Genera una rete con connessioni completamente caotiche
	void caoticNet() {} // TODO da sviluppare
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////FUNZIONI DI ADDESTRAMENTO//////////////////////////////////////////////////////
	//la funzione considera i parametri della rete mlp associta per addestrare la memoria associativa 
	void trainSupportNet(float Wi, float We, int g) {
		float infl = 0, BPerr = 0;
		float Dinfl = 0, DbpErr = 0;
		for (int i = 0; i < mlp->nLayers; i++) {
			for (int j = 0; j < mlp->numNeurons(i); j++) {
				for (int k = 0; k < mlp->numCon(i, j); k++) {
					BPerr = mlp->getTarget(i, j)->influenceOutput[k];
					infl = mlp->getTarget(i, j)->OutArcs[k].target->influenceInput[mlp->idBaseConReftoTargetInfl(mlp->getTarget(i, j), k)];
					Dinfl = DeltaHW(Wi, infl, g);
					DbpErr = DeltaHW(We, BPerr, g);
					binds[searchFromTarget(mlp->getTarget(i, j))].base->OutArcs[k].weight += Dinfl + DbpErr;
				}
			}
		}
	}
	//funzione che taglia gli archi con il peso sotto una certa soglia 
	void cutMadArc(int maxCut, int alfa) {
		//!!!la procedura supporta soltanto reti ad anello su un solo layer!!!
		//maxCut rappresenta il numero massimo di rami da tagliare in questa esecuzione
		//alfa rappresenta il coefficiente di soglia percentuale sotto il quale il ramo puo essere tagliato (la soglia è una percentuale)
		float Wsum = 0;
		u_int Wn = 0;
		float Wmax = std::numeric_limits<float>::min();
		float Wmin = std::numeric_limits<float>::max();
		for (int i = 0; i < numNeurons(0); i++) {
			for (int j = 0; j < numCon(0, i); j++) {
				Wsum += getTarget(0, i)->OutArcs[j].weight;
				if (getTarget(0, i)->OutArcs[j].weight > Wmax) { Wmax = getTarget(0, i)->OutArcs[j].weight; }
				if (getTarget(0, i)->OutArcs[j].weight < Wmin) { Wmax = getTarget(0, i)->OutArcs[j].weight; }
				Wn++;
			}
		}
		nArc = Wn;
		mlp->nArc = nArc;
		float Wmean = Wsum / Wn;
		float treashold = ((Wmax - Wmin)*alfa) + Wmin; // alfa = 0 -> Treashold = Wmin , alfa = 1 -> Treashold = Wmax
		int Tcol = 0, TmlpL = 0, TmlpC = 0, mlpL = 0, mlpC = 0;
		vector <u_int> acs = casualVector(nNeurons - mlp->numNeurons(mlp->nLayers - 1));

		for (int i = 0; i < acs.size(); i++) {
			for (int j = 0; j < getTarget(0, acs[i])->numOutArcs; j++) {
				if (getTarget(0, acs[i])->OutArcs[j].weight < treashold) { // condizione sufficiente all'eliminazione

					Tcol = getTarget(0, acs[i])->OutArcs[j].target->column; //colonna del neurone Hopfield da eliminare
					TmlpL = binds[serchFromBase(getTarget(0, acs[i])->OutArcs[j].target)].target->layer; // numLayer del neurone mlp da eliminare
					TmlpC = binds[serchFromBase(getTarget(0, acs[i])->OutArcs[j].target)].target->column; // numCol del neurone mlp da eliminare
					mlpL = binds[serchFromBase(getTarget(0, acs[i]))].target->layer; // numLayer del neurone mlp contenente la connessione
					mlpC = binds[serchFromBase(getTarget(0, acs[i]))].target->column; // numCol del neurone mlp contenente la connessione

					// procedura di eliminazione arco
					if (getTarget(0, acs[i])->OutArcs[j].target->numInArcs == 1) { // se il neurone obbiettivo ha solo l'arco da eliminare elimino anche il corrispondente elemento bind
						//binds.erase(binds.begin() + serchFromBase(getTarget(0, acs[i])->OutArcs[j].target)); //cerco e elimino l'elemento bind dei due neuroni
						//deleteArc(0, acs[i], 0, Tcol); // elimino l'arco nella rete Hopfield
						//mlp->deleteArc(mlpL, mlpC, TmlpL, TmlpC); //elimino l'arco nella rete mlp

						//TODO utilizza funzioni apposite per l'eliminazione dei neuroni delle due reti

						nArc++;
					}
					else { //altrimenti elimino semplicemente l'arco nelle due reti
						deleteArc(0, acs[i], 0, Tcol); // elimino l'arco nella rete Hopfield
						mlp->deleteArc(mlpL, mlpC, TmlpL, TmlpC); //elimino l'arco nella rete mlp
					}
					nArc--;
					mlp->nArc = nArc;
					maxCut--;
					if (maxCut < 1) return;
				}
			}

		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////FUNZIONI MODIFICA RETE///////////////////////////////////////////////////////////
	void HdeleteArc(int Nlayer, int Ncolumn, int targetLayer, int targetColumn) {
		for (int i = 0; i < Layers[Nlayer].Neurons[Ncolumn].numOutArcs; i++) {
			if (Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->layer == targetLayer && Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->column == targetColumn) {

				//modifico i parametri della rete relazionati a tale arco
				Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->numInArcs--;
				if (Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->influenceInput.size() > 0)Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->influenceInput.pop_back();
				if (Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->influenceOutput.size() > 0)Layers[Nlayer].Neurons[Ncolumn].influenceOutput.pop_back();

				if (Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->numInArcs == 0 && Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->layer != 0) {
					deleteNeuron(Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->layer, Layers[Nlayer].Neurons[Ncolumn].OutArcs[i].target->column);
				}
				else {
					Layers[Nlayer].Neurons[Ncolumn].OutArcs.erase(Layers[Nlayer].Neurons[Ncolumn].OutArcs.begin() + i);
					Layers[Nlayer].Neurons[Ncolumn].numOutArcs--;
				}

				return;
			}
		}
		cout << "errore: arco (" << Nlayer << "-" << Ncolumn << ") -> (" << targetLayer << "-" << targetColumn << ") non trovato, eliinazione fallita" << endl;
	}
	void HdeleteArcByRef(int Nlayer, int Ncolumn, arc *targetArc) { //elimina un arco con gli indici della base e il puntatore alla connsessione

																   //modifico i parametri della rete relazionati a tale arco
		targetArc->target->numInArcs--;
		if (targetArc->target->influenceInput.size() > 0)targetArc->target->influenceInput.pop_back();
		if (Layers[Nlayer].Neurons[Ncolumn].influenceOutput.size() > 0)Layers[Nlayer].Neurons[Ncolumn].influenceOutput.pop_back();

		if (targetArc->target->numInArcs == 0) {
			deleteNeuron(targetArc->target->layer, targetArc->target->column);
		}
		else {
			Layers[Nlayer].Neurons[Ncolumn].OutArcs.erase(Layers[Nlayer].Neurons[Ncolumn].OutArcs.begin() + getOutConTargetID(getTarget(Nlayer, Ncolumn), targetArc->target));
			Layers[Nlayer].Neurons[Ncolumn].numOutArcs--;
		}

	}
	void HdeleteNeuron(int Nlayer, int Ncolumn) {
		// elimino l'arco dai neuroni che puntano al nodo da eliminare e
		// decremento il numero di archi in output hai neuroni con come target il neurone da eliminare
		for (int i = 0; i < Nlayer; i++) {
			for (int j = 0; j < Layers[i].numNeurons; j++) {
				for (int k = 0; k < Layers[i].Neurons[j].numOutArcs; k++) {
					if (Layers[i].Neurons[j].OutArcs[k].target->layer == Nlayer && Layers[i].Neurons[j].OutArcs[k].target->column == Ncolumn) {
						deleteArcByRef(i, j, &Layers[i].Neurons[j].OutArcs[k]);
					}
				}
			}
		}
		// eliminio gli archi del neurone da eliminare
		for (int i = 0; i < Layers[Nlayer].Neurons[Ncolumn].numOutArcs; i++) {
			deleteArcByRef(Nlayer, Ncolumn, &Layers[Nlayer].Neurons[Ncolumn].OutArcs[i]);
		}

		// decremento l'indice colonna dei neuroni successivi nello stesso layer
		for (int i = Ncolumn + 1; i < Layers[Nlayer].numNeurons; i++) Layers[Nlayer].Neurons[i].column--;

		// elimino il neurone
		Layers[Nlayer].Neurons.erase(Layers[Nlayer].Neurons.begin() + Ncolumn);
		Layers[Nlayer].numNeurons--;
		nNeurons--;
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////ALTRE FUNZIONI/////////////////////////////////////////////////////////////////
	int serchFromBase(ptNeuron base) {
		for (int i = 0; i < binds.size(); i++) {
			if (base == binds[i].base) {
				return i;
			}
		}
		return -1;
	}
	int searchFromTarget(ptNeuron target) {
		for (int i = 0; i < binds.size(); i++) {
			if (target == binds[i].target) {
				return i;
			}
		}
		return -1;
	}
	//funzione per il condizionamento dei valori dei vettori di retropropagazione
	float DeltaHW(float W, float x, float g) {
		return W * powf((2 * x - 1), (1 + 2 * g));
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

class StructuralLearning {
private:
	MLP * mlp; // rete mlp
	Hopfield* hpd; // rete Hopfield

public:

	StructuralLearning(MLP* ptMlp, Hopfield* ptHpd) { //costruttore
		mlp = ptMlp;
		hpd = ptHpd;
		mlp->initVectorsProfiler();
	}

	void StructuralBP(int iter, float eps, float beta, float ErrPercent, float Wi, float We, int g, int cutStart, float alfa, float SmaxErr, float maxWloss) {
		int nWt0 = mlp->nArc; // numero dei pesi al tempo t0
		int nWtn = 0; // numero dei pesi rimasti all'i-esima iterazzione
		float WdelP = 0; // percentuale dei pesi eliminati dall'inizio dell'addestramento
		float err = 0; // errore del singolo neurone
		float Err = 0; // errore quadratio del singolo esempio
		float Gerr = 0; // errore quadratico dell'intero dataset
		float Perr = 0;
		float delta = 0;
		long long t0 = 0;
		long long t1 = 0;
		long long inT1 = 0;
		double elapsedMilliseconds = 0;
		double executionTime = 0;
		double inputTime = 0;
		int x = 0;
		vector<float> Out(mlp->numNeurons(mlp->nLayers - 1)); // vettore supporto output rete
		vector<float> GerrStory;

		for (int t = 0; t < iter; t++) {
			Gerr = 0;
			Perr = 0;
			for (int e = 0; e < mlp->examples.size(); e++) {
				t0 = mlp->PerformanceCounter();
				// eseguo l'esempio
				mlp->inputNetProfiler(mlp->examples[e].input, Out);

				t1 = mlp->PerformanceCounter();
				elapsedMilliseconds = ((t1 - t0) * 1000.0) / mlp->PerformanceFrequency();
				inputTime += elapsedMilliseconds;

				// clacolo il delta delle variazioni dei pesi dello strato di output
				Err = 0;
				for (int r = 0; r < Out.size(); r++) {
					delta = mlp->map[r].maxValue - mlp->map[r].minValue;
					err = Out[r] - mlp->examples[e].Doutput[r];
					Perr += abs(err / mlp->examples[e].Doutput[r]) * 100; // errore percentuale
					Err += pow(err, 2);


					if (x == e && t > -1) {


						cout << endl << "es." << x << " Y" << r << " = " << Out[r] << "   " << "D" << r << " = " << mlp->examples[e].Doutput[r] << "  err: " << abs(Out[r] - mlp->examples[e].Doutput[r]) << endl;

						if (r == Out.size() - 1) {
							x--;
							mlp->WeightsStamp("fc");
							/*mlp->stampInputInfluences(true);
							mlp->stampOutputErrorPropagation(true);*/ //erroi retropropagati nell'iterazione precedente
							/*Sleep(5000);*/
						}
						if (x < 0) x = mlp->examples.size() - 1;
					}

					mlp->resetAbsSumDelta();

					//mErr = ((err- map[r].minValue)/ delta) ;
					// calcolo il delta della variazione dei pesi
					mlp->Layers[mlp->nLayers - 1].Neurons[r].BPerr = mlp->reverseMap(r, err) * (mlp->reverseMap(r, Out[r]) *(1 - mlp->reverseMap(r, Out[r])));
					//applico le correzioni ai bayes
					mlp->Layers[mlp->nLayers - 1].Neurons[r].oldBayesDelta = (-eps * (mlp->Layers[mlp->nLayers - 1].Neurons[r].BPerr)) + beta * mlp->Layers[mlp->nLayers - 1].Neurons[r].oldBayesDelta;
					mlp->Layers[mlp->nLayers - 1].Neurons[r].bayes += mlp->Layers[mlp->nLayers - 1].Neurons[r].oldBayesDelta;

				}
				//applico le correzioni ai pesi dello strato di output
				for (int i = 0; i < mlp->numNeurons(mlp->nLayers - 2); i++) {
					for (int j = 0; j < mlp->numCon(mlp->nLayers - 2, i); j++) {

						mlp->Layers[mlp->nLayers - 2].Neurons[i].OutArcs[j].oldDelta = (-eps * (mlp->Layers[mlp->nLayers - 2].Neurons[i].OutArcs[j].target->BPerr) * mlp->Layers[mlp->nLayers - 2].Neurons[i].output) + beta * mlp->Layers[mlp->nLayers - 2].Neurons[i].OutArcs[j].oldDelta;
						mlp->Layers[mlp->nLayers - 2].Neurons[i].OutArcs[j].weight += mlp->Layers[mlp->nLayers - 2].Neurons[i].OutArcs[j].oldDelta;
						mlp->Layers[mlp->nLayers - 2].Neurons[i].absDeltaSum += abs(mlp->Layers[mlp->nLayers - 2].Neurons[i].OutArcs[j].oldDelta);

					}
				}
				// rieseguo la procedura per tutti gli altri strati
				for (int i = mlp->nLayers - 2; i > 0; i--) { // dal penultimo strato al secondo
															 // clacolo il delta delle variazioni dei pesi dello strato i-1
					for (int j = 0; j < mlp->numNeurons(i); j++) {
						err = 0;
						for (int k = 0; k < mlp->numCon(i, j); k++) err += mlp->Layers[i].Neurons[j].OutArcs[k].target->BPerr * mlp->Layers[i].Neurons[j].OutArcs[k].weight;
						mlp->Layers[i].Neurons[j].BPerr = mlp->DsigOut(i, j)*err;
						//applico le correzioni ai bayes
						mlp->Layers[i].Neurons[j].oldBayesDelta = (-eps * (mlp->Layers[i].Neurons[j].BPerr)) + beta * mlp->Layers[i].Neurons[j].oldBayesDelta;
						mlp->Layers[i].Neurons[j].bayes += mlp->Layers[i].Neurons[j].oldBayesDelta;

					}
					// applico le correzioni ai pesi dello strato i-1
					for (int j = 0; j < mlp->numNeurons(i - 1); j++) {
						for (int k = 0; k < mlp->numCon(i - 1, j); k++) {

							mlp->Layers[i - 1].Neurons[j].OutArcs[k].oldDelta = (-eps * (mlp->Layers[i - 1].Neurons[j].OutArcs[k].target->BPerr) * mlp->Layers[i - 1].Neurons[j].output) + beta * mlp->Layers[i - 1].Neurons[j].OutArcs[k].oldDelta;
							mlp->Layers[i - 1].Neurons[j].OutArcs[k].weight += mlp->Layers[i - 1].Neurons[j].OutArcs[k].oldDelta;
							mlp->Layers[i - 1].Neurons[j].absDeltaSum += abs(mlp->Layers[i - 1].Neurons[j].OutArcs[k].oldDelta);
						}
					}
				}

				mlp->resetVectorsProfiler(false, true);

				//calcolo dei vettori di influenza dell'errore retropropagato da ogni output
				for (int i = mlp->nLayers - 2; i >= 0; i--) {
					for (int j = 0; j < mlp->numNeurons(i); j++) {
						for (int k = 0; k < mlp->numCon(i, j); k++) {
							mlp->getTarget(i, j)->influenceOutput[k] += abs((mlp->getTarget(i, j)->OutArcs[k].oldDelta) / (mlp->getTarget(i, j)->absDeltaSum));
						}
					}
				}

				///////////////////////////////////sezione dell'algoritmo in cui la rete ha i vettori di propagazione caricati/////////////////////////

				hpd->trainSupportNet(Wi, We, g);

				///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

				Gerr += Err;
				mlp->resetPotential(); // azzero i potenziali di output della rete
				mlp->resetAbsSumPotenzial(); //azzero le sommatorie in valore assoluto dei contributi delle connessioni nell'output del neurone

				t1 = mlp->PerformanceCounter();
				elapsedMilliseconds = ((t1 - t0) * 1000.0) / mlp->PerformanceFrequency();
				executionTime += elapsedMilliseconds;
			}

			///////////////////////////////////// procedura di eliminazione archi ////////////////////////////////////////////////////

			if (t > cutStart) {
				if (t == cutStart + 1) nWt0 = mlp->nArc;
				WdelP = 1.0f - (mlp->nArc / (float)nWt0);
				WdelP = WdelP * 100.0f;
				if (mlp->NetErrPercent > SmaxErr || WdelP >= maxWloss) {
					if (mlp->NetErrPercent > SmaxErr) cout << "error reached: " << mlp->NetErrPercent << endl;
					if (WdelP >= maxWloss) cout << "weights lossed: " << WdelP << "%" << endl;
					return;
				}
				hpd->cutMadArc(1, alfa);
			}

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			Gerr /= 2;
			Perr /= mlp->examples.size()*Out.size();
			executionTime /= mlp->examples.size();
			inputTime /= mlp->examples.size();
			mlp->NetPerformance = inputTime;
			mlp->NetErrPercent = Perr;



			cout << "Iterazione  " << t << "   errore quadratico: " << Gerr << "   errore percentuale medio:  " << Perr << "  %err" << endl
				<< "exemple time: " << executionTime << " ms  inputTime: " << inputTime << " ms" << endl
				<< "Net integrity:" << (mlp->nArc / (float)nWt0) * 100 << "%    arcs:  " << mlp->nArc << endl;

			//if (t > 100 && t < 200)Sleep(500);
			if (Perr < ErrPercent) {
				cout << "Percentuale di errore obbiettivo raggiunta!" << endl;
				return;
			}

		}
	}

};

////////////////////////////////////////////////////////////////CUDA Kernels//////////////////////////////////////////////////////////

//resetta il valore di una variabile all'interno della scheda grafica
__global__ void CUDAresetVar(float *val) {
	*val = 0;
}
//applica ad ogni arco della rete la correzione del peso
__global__ void CUDAapplyWeightCorrections(float eps, float *NeuronOut, float *BPerr, float *weights, int *ArcIn, int *ArcOut, int nArcs) {
	unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < nArcs) {
		weights[i] += -eps * BPerr[ArcIn[i]] * NeuronOut[ArcOut[i]];
	}
}
__global__ void CUDAapplyBayesCorrections(float eps, float *BPerr, float *Bayes, int startN, int endN) {
	unsigned int i = startN + (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i <= endN) {
		Bayes[i] += -eps * BPerr[i];
	}
}
//applica alla sommatoria degli errori pesati e retropropagati ad ogni neurone la derivata puntuale della sigmoide 
//DEPRECATED!!!
/*__global__ void CUDAapplayDsigToBPerr(float *NeuronOut, float *BPerr, int nNeuron) {
	unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < nNeuron) {
		BPerr[i] *= NeuronOut[i] * (1 - NeuronOut[i]);
	}
}*/
//retropropaga l'errore nella rete 
__global__ void CUDAPropagationErr(float *BPerr, float *weights, float *NeuronOut, int *ArcIn, int *ArcOut, int startA, int endA) {
	unsigned int i = startA + (blockIdx.x * blockDim.x) + threadIdx.x;

	//retropropago l'errore dai neuroni successivi
	if (i <= endA) {
		//BPerr[ArcOut[i]] += BPerr[ArcIn[i]] * weights[i];
		atomicAdd(&BPerr[ArcOut[i]], BPerr[ArcIn[i]] * weights[i]);
	}
}
__global__ void CUDAoutDiff(float *BPerr, float *NeuronOut, int startN, int endN) {
	int i = startN + (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i <= endN) {
		BPerr[i] *= NeuronOut[i] * (1 - NeuronOut[i]);
	}
}
//calcola l'errore dei neuroni dello strato output
__global__ void CUDAoutputErr(float *NeuronOut, int OutputRef, int numNeurons, int inputN, float *BPerr, float *examples, int exampleRef, float *mapMaxOut, float *mapMinOut, float *MeanErr) {
	unsigned int i = (OutputRef) + (blockIdx.x * blockDim.x) + threadIdx.x; //indice di scorrimento vettori: NeuronOut, BPerr, 
	unsigned int e = (exampleRef + inputN) + (blockIdx.x * blockDim.x) + threadIdx.x; //indice di scorrimento vettori: examples
	unsigned int m = (blockIdx.x * blockDim.x) + threadIdx.x; // indice di scorrimento vettori: mapMaxOut, mapMinOut
	//if (i == 0) *MeanErr = 0;
	if (i < numNeurons) {

		float delta = mapMaxOut[m] - mapMinOut[m];
		BPerr[i] = (NeuronOut[i] - ((examples[e] - mapMinOut[m]) / delta)) * NeuronOut[i] * (1 - NeuronOut[i]); // formula valida solo per i neuroni di uscita
		//atomicAdd(MeanErr, (abs((((NeuronOut[i] * delta) + mapMinOut[m]) - examples[e]) / examples[e]))*100.0f);
		atomicAdd(MeanErr, abs((((NeuronOut[i] * delta) + mapMinOut[m]) - examples[e])/ examples[e]) * 100.0f);
		//*MeanErr += abs((examples[e] - ((NeuronOut[i] * delta) + mapMinOut[m])) / examples[e]); // calcolo l'errore percentuale sulla singola uscita e lo sommo 
	}
}
//resetta un dato vettore 
__global__ void CUDAresetVector(float *vect, int size) {
	unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < size) vect[i] = 0.0f;
}
//imposta i valori di output dei neuroni di input al valore dell'esempio
__global__ void CUDAsetInput(float *NeuronOut, int inputN, int exampleRef, float *example) {
	unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < inputN)NeuronOut[i] = example[exampleRef + i];
}
__global__ void CUDAsetSingleInput(float *NeuronOut, int inputN, float *example) {
	unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < inputN)NeuronOut[i] = example[i];
}
//applica la sigmoide ai potenziali dei neuroni in un dato intervallo
__global__ void CUDAsigLayer(float *NeuronOut, int start, int end) {
	unsigned int i = start + (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i <= end) {
		NeuronOut[i] = 1 / (1 + expf(-NeuronOut[i]));
	}
}
//aggiunge all'output del neurone il contributo del bayes
__global__ void CUDAbayesInput(float *NeuronOut, float *Bayes, int start, int end) {
	unsigned int i = start + (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i <= end) {
		NeuronOut[i] += Bayes[i];
	}
}
//propaga l'informazione dai neuroni dello strato input a quello di output
//TODO sostituire l'utilizzo di atomicAdd con la riduzione delle somme (utilizando atomicAdd il minor numero possibile di volte per ogni neurone)
__global__ void CUDAlayerInput(float *weights, int *ArcIn, int *ArcOut, float *NeuronOut, int start, int end) {
	unsigned int i = start + (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i <= end) {
		atomicAdd(&NeuronOut[ArcIn[i]], NeuronOut[ArcOut[i]] * weights[i]); //addizione bloccante non permette ad altri thread di sovrascrivere il falore finche l'operazione non è completata
		//printf("Neurone %d ( %f ) += Neuron %d ( %f ) * peso ( %f ) \n", ArcIn[i], NeuronOut[ArcIn[i]], ArcOut[i], NeuronOut[ArcOut[i]], weights[i]);
		//NeuronOut[ArcIn[i]] += NeuronOut[ArcOut[i]] * weights[i];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CUDAcore {
	//api di interfacciamento alla GPU
private:

public:
	//TODO aggiungere il vettore dei bayes e relativa funzione di applicazione e correzione

	cudaDeviceProp prop; //Device specs struct
	int GpuID = 0;

	//struttura contenente i puntatori alle aree di memoria conenenti i parametri della rete nella GPU
	struct devNetParams {
		float *weights = 0;
		int *ArcIn = 0;
		int *ArcOut = 0;
		float *examples = 0;
		float *NeuronOut = 0;
		float *Bayes = 0;
		float *BPerr = 0;
		float *mapMaxOut = 0;
		float *mapMinOut = 0;
		int *NeurInLyr = 0;
		int *priority = 0;
		float *MeanErr = 0;
		float *InputRT = 0;
	}gpuNetParams;

	vector<float> weights; //pei della rete
	vector<int> ArcIn; //target dell'n-esimo arco
	vector<int> ArcOut; //base dell'n-esimo arco
	vector<float> NeuronOut; //vettore contenente l'output dei neuroni
	vector<float> Bayes; //vettore contenente i bayes dei neuroni
	vector<float> BPerr; // vettore contenete gli errori retropropagati
	vector<float> mapMaxOut; //vettore contenente il massimo valore degli output
	vector<float> mapMinOut; //vettore contenente il minimo valore degli output
	vector<int> priority; // vettore contenente i punti di sincronizazione dei thread
	vector<int> NeurInLyr; //vettore contenente gli indici dell'ultimo neurone di ogni layer
	vector<float>examples; //vettore degli esempi
	float MeanErr = 0; //veriabile contenente l'errore medio percentuale della rete
	int inputN, outputN; //passo di esecuzione elementi del vettore esempi

	CUDAcore(int nGpu) {
		GpuID = nGpu;
		checkCuda(cudaGetDeviceProperties(&prop, nGpu)); // carica lo struct cudaDeviceProp prop con le caratteristiche della GPU con indice 0
	}
	/*per convertire gli oggetti vector in Array
	std::vector<double> v;
	double* a = &v[0];
	*/

	void cudaNetCopyMLP(MLP *pt) {
		cout << "copying the net into CUDAcore.." << endl;
		weights.resize(pt->nArc);
		ArcIn.resize(pt->nArc);
		ArcOut.resize(pt->nArc);
		NeuronOut.resize(pt->nNeurons);
		Bayes.resize(pt->nNeurons);
		BPerr.resize(pt->nNeurons);
		priority.resize(pt->nLayers + 1);
		NeurInLyr.resize(pt->nLayers + 1);
		mapMaxOut.resize(pt->map.size());
		mapMinOut.resize(pt->map.size());
		inputN = pt->numNeurons(0);
		outputN = pt->numNeurons(pt->nLayers - 1);

		int NeuronIdx = 0;
		int ArcIdx = 0;
		vector<int> neurons(pt->nLayers);
		//carico il vettore di mappatura dell'output della rete
		for (int i = 0; i < pt->map.size(); i++) {
			mapMaxOut[i] = pt->map[i].maxValue;
			mapMinOut[i] = pt->map[i].minValue;
		}

		NeurInLyr[0] = -1; // setto il primo valore 
		priority[0] = -1; // setto il primo valore 

		//carico i parametri della rete
		for (int i = 0; i < pt->nLayers; i++) {

			for (int j = 0; j < pt->numNeurons(i); j++) {

				Bayes[NeuronIdx] = pt->getTarget(i, j)->bayes;

				for (int k = 0; k < pt->numCon(i, j); k++) {

					weights[ArcIdx] = pt->getTarget(i, j)->OutArcs[k].weight;
					ArcIn[ArcIdx] = pt->getTarget(i, j)->OutArcs[k].target->neurIdx;
					ArcOut[ArcIdx] = pt->getTarget(i, j)->neurIdx;
					ArcIdx++;
				}
				NeuronIdx++;
			}
			NeurInLyr[i + 1] = NeuronIdx - 1; // salvo l'indice dell'ultimo neurone del layer corrente
			priority[i + 1] = ArcIdx - 1; // salvo l'indice dell'ultimo arco del layer corrente
		}
	}

	void cudaNetPasteMLP(MLP *pt) {
		int idx = 0;
		int Nidx = 0;
		for (int i = 0; i < pt->nLayers; i++) {
			for (int j = 0; j < pt->numNeurons(i); j++) {
				pt->getTarget(i, j)->bayes = Bayes[Nidx++];
				for (int k = 0; k < pt->numCon(i, j); k++) {
					pt->getTarget(i, j)->OutArcs[k].weight = weights[idx++];
				}
			}
		}
	}

	void cudaNetCopyHopfield(Hopfield* pt) {

	}

	void cudaNetCopyExamples(MLP *pt) {
		cout << "copying example into CUDAcore" << endl;
		examples.resize(pt->examples.size()*(pt->numNeurons(0) + pt->numNeurons(pt->nLayers - 1)));
		int idx = 0;
		for (int i = 0; i < pt->examples.size(); i++) {
			for (int j = 0; j < pt->numNeurons(0); j++) {
				examples[idx++] = pt->examples[i].input[j];

			}
			for (int j = 0; j < pt->numNeurons(pt->nLayers - 1); j++) {
				examples[idx++] = pt->examples[i].Doutput[j];
			}
		}
	}
	////////////////////////////////////////////////////////////////CUDA Kernel functions/////////////////////////////////////////////////

	//esegue le operazioni di allocamento memoria e preparazione al lancio del kernel di propagazione della rete
	cudaError_t hostCUDAtrainingNet(float eps, int Niter, int ThxBlock) {
		cout << "learning is started!" << endl;
		//host variables
		float *Cweights = &weights[0];
		int *CArcIn = &ArcIn[0];
		int *CArcOut = &ArcOut[0];
		float *CNeuronOut = &NeuronOut[0];
		float *CBayes = &Bayes[0];
		float *CBPerr = &BPerr[0];
		float *CmapMaxOut = &mapMaxOut[0];
		float *CmapMinOut = &mapMinOut[0];
		float *Cexamples = &examples[0];
		int *CNeurInLyr = &NeurInLyr[0];
		int *Cpriority = &priority[0];
		float *CMeanErr = &MeanErr;

		//device variables
		float *dev_weights = 0;
		int *dev_ArcIn = 0;
		int *dev_ArcOut = 0;
		float *dev_examples = 0;
		float *dev_NeuronOut = 0;
		float *dev_Bayes = 0;
		float *dev_BPerr = 0;
		float *dev_mapMaxOut = 0;
		float *dev_mapMinOut = 0;
		int *dev_NeurInLyr = 0;
		int *dev_priority = 0;
		float *dev_MeanErr = 0;

		//int ThxBlock = 1024;

		cudaError_t cudaStatus;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(GpuID);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;

		// Allocate GPU buffers for vectors    
		cudaStatus = cudaMalloc((void**)&dev_weights, weights.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_ArcIn, ArcIn.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_ArcOut, ArcOut.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_NeuronOut, NeuronOut.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_Bayes, Bayes.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_BPerr, BPerr.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_mapMaxOut, mapMaxOut.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_mapMinOut, mapMinOut.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_examples, examples.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_NeurInLyr, NeurInLyr.size() * sizeof(int));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_priority, priority.size() * sizeof(int));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&dev_MeanErr, sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;


		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_weights, Cweights, weights.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_ArcIn, CArcIn, ArcIn.size() * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_ArcOut, CArcOut, ArcOut.size() * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_NeuronOut, CNeuronOut, NeuronOut.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_Bayes, CBayes, Bayes.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_BPerr, CBPerr, BPerr.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_mapMaxOut, CmapMaxOut, mapMaxOut.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_mapMinOut, CmapMinOut, mapMinOut.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_examples, Cexamples, examples.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_NeurInLyr, CNeurInLyr, NeurInLyr.size() * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_priority, Cpriority, priority.size() * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(dev_MeanErr, CMeanErr, sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;



		//////////////////lancio dei kernel all'interno della gpu////////////////
		int startA = 0;
		int endA = 0;
		int startN = 0;
		int endN = 0;
		int numLayerArcs = 0;
		int numLayerNeur = 0;
		int numOfBlocksMax = 0;
		int numOfBlocksA = 0;
		int numOfBlocksN = 0;
		int numOfBlocksOut = floorf(outputN / ThxBlock) + 1;
		int exampleRef = 0;
		int outputRef = NeuronOut.size() - outputN;
		long long t0 = 0, t1 = 0;
		long long t0in = 0, t1in = 0;
		double elapsedMilliseconds = 0;
		double elapsedInMilliseconds = 0;

		//debug////////////
		int en = 0;
		float delta = 0;
		//////////////////

		for (int it = 0; it < Niter; it++) { //scorro le iterazioni

			t0 = PerformanceCounter();

			for (int t = 0; t < (examples.size() / (inputN + outputN)); t++) { //scorro gli esempi
				//imposto il riferimento per l'esempio di input
				exampleRef = t * (inputN + outputN);
				t0in = PerformanceCounter();
				//resetto il vettore contenente lo stato di attivazione dei neuroni
				numOfBlocksA = (floorf(NeuronOut.size() / ThxBlock) + 1);
				CUDAresetVector << <numOfBlocksA, ThxBlock >> > (dev_NeuronOut, NeuronOut.size());

				//////////////////////////////DEBUG///////////////////////////////
				//cudaStatus = cudaMemcpy(CNeuronOut, dev_NeuronOut, NeuronOut.size() * sizeof(float), cudaMemcpyDeviceToHost);
				//if (cudaCheckStatus(cudaStatus) == true) goto Error;
				//copy(CNeuronOut, CNeuronOut + NeuronOut.size(), NeuronOut.begin());
				///////////////////////////////////////////////////////////////////

				//imposto i valori di input ai neuroni dello strato input
				numOfBlocksA = (floorf(inputN / ThxBlock) + 1);
				CUDAsetInput << <numOfBlocksA, ThxBlock >> > (dev_NeuronOut, inputN, exampleRef, dev_examples);

				//////////////////////////////DEBUG///////////////////////////////
				//cudaStatus = cudaMemcpy(CNeuronOut, dev_NeuronOut, NeuronOut.size() * sizeof(float), cudaMemcpyDeviceToHost);
				//if (cudaCheckStatus(cudaStatus) == true) goto Error;
				//copy(CNeuronOut, CNeuronOut + NeuronOut.size(), NeuronOut.begin());
				///////////////////////////////////////////////////////////////////

				//propagazione dell'input nella rete

				startA = 0; // indice di partenza dei vettori archi
				endA = 0; // ultimo indice dei vettori archi
				startN = 0; // indice di partenza dei vettori neuroni
				endN = 0; // ultimo indice dei vettori neuroni

				for (int i = 0; i < priority.size() - 1; i++) { //NB non viene applicata la sigmoide allo strato di input eventulmente correggi

					startA = priority[i] + 1;
					endA = priority[i + 1];

					if (i < priority.size() - 2) {
						startN = NeurInLyr[i + 1] + 1;
						endN = NeurInLyr[i + 2];
					}

					numLayerArcs = endA - startA + 1;
					numLayerNeur = endN - startN + 1;

					numOfBlocksA = floorf(numLayerArcs / ThxBlock) + 1;
					numOfBlocksN = floorf(numLayerNeur / ThxBlock) + 1;

					if (i < priority.size() - 2) {
						CUDAlayerInput << <numOfBlocksA, ThxBlock >> > (dev_weights, dev_ArcIn, dev_ArcOut, dev_NeuronOut, startA, endA); //propago l'output dei neuroni al prossimo/i layer
						CUDAbayesInput << < numOfBlocksN, ThxBlock >> > (dev_NeuronOut, dev_Bayes, startN, endN); //applico il contributo dei bayes all output dei neuroni del layer corrente 
						CUDAsigLayer << <numOfBlocksN, ThxBlock >> > (dev_NeuronOut, startN, endN); //applico la sigmoide allo stato di attivazione dei neuroni
						//////////////////////////////DEBUG/////////////////////////////////
						cudaStatus = cudaMemcpy(CNeuronOut, dev_NeuronOut, NeuronOut.size() * sizeof(float), cudaMemcpyDeviceToHost);
						if (cudaCheckStatus(cudaStatus) == true) goto Error;
						copy(CNeuronOut + outputRef, CNeuronOut + NeuronOut.size(), NeuronOut.begin() + outputRef);
						///////////////////////////////////////////////////////////////////
					}
				}

				t1in = PerformanceCounter();
				elapsedInMilliseconds += ((t1in - t0in) * 1000.0) / PerformanceFrequency();


				//resetto il vettore contenente l'errore  dei neuroni
				numOfBlocksN = (floorf(BPerr.size() / ThxBlock) + 1);
				CUDAresetVector << <numOfBlocksN, ThxBlock >> > (dev_BPerr, BPerr.size());

				//////////////////////////////DEBUG///////////////////////////////
				//cudaStatus = cudaMemcpy(CBPerr, dev_BPerr, BPerr.size() * sizeof(float), cudaMemcpyDeviceToHost);
				//if (cudaCheckStatus(cudaStatus) == true) goto Error;
				//copy(CBPerr, CBPerr + BPerr.size(), BPerr.begin());
				///////////////////////////////////////////////////////////////////

				CUDAresetVar <<<1, 1 >>> (dev_MeanErr);
				CUDAoutputErr << <numOfBlocksOut, ThxBlock >> > (dev_NeuronOut, outputRef, NeuronOut.size(), inputN, dev_BPerr, dev_examples, exampleRef, dev_mapMaxOut, dev_mapMinOut, dev_MeanErr);
				cudaMemcpy(CMeanErr, dev_MeanErr, sizeof(float), cudaMemcpyDeviceToHost);
				//////////////////////////////DEBUG///////////////////////////////
				//cudaStatus = cudaMemcpy(CBPerr, dev_BPerr, BPerr.size() * sizeof(float), cudaMemcpyDeviceToHost);
				//if (cudaCheckStatus(cudaStatus) == true) goto Error;
				//copy(CBPerr, CBPerr + BPerr.size(), BPerr.begin());

				///////////////////////////////////////////////////////////////////

				////////////////////////////////////visualizzazione dell'esempio///////////////////////////////////
				if (en == t) {

					cudaStatus = cudaMemcpy(CNeuronOut, dev_NeuronOut, NeuronOut.size() * sizeof(float), cudaMemcpyDeviceToHost);
					if (cudaCheckStatus(cudaStatus) == true) goto Error;
					copy(CNeuronOut, CNeuronOut + NeuronOut.size(), NeuronOut.begin());

					cout << "esempio " << en << endl;
					for (int on = 0; on < outputN; on++) {
						delta = mapMaxOut[on] - mapMinOut[on];
						cout << "Y" << on << ": " << (NeuronOut[NeuronOut.size() - outputN + on] * delta) + mapMinOut[on] << "   D" << on << ": " << examples[exampleRef + inputN + on] << endl;
					}
					cout << endl;
					en--;
					if (en < 0)en = (examples.size() / (inputN + outputN)) - 1;
				}
				///////////////////////////////////////////////////////////////////////////////////////////////////

				MeanErr += *CMeanErr / outputN;


				//retropropagazione dell'errore

				for (int i = priority.size() - 2; i > 1; i--) {

					startA = priority[i - 1] + 1;
					endA = priority[i];
					startN = NeurInLyr[i - 1] + 1;
					endN = NeurInLyr[i];

					numLayerArcs = endA - startA + 1;
					numLayerNeur = endN - startN + 1;

					numOfBlocksA = floorf(numLayerArcs / ThxBlock) + 1;
					numOfBlocksN = floorf(numLayerNeur / ThxBlock) + 1;
					//numOfBlocksMax = maxOf(numOfBlocksA, numOfBlocksN);

					CUDAPropagationErr <<<numOfBlocksA, ThxBlock >>> (dev_BPerr, dev_weights, dev_NeuronOut, dev_ArcIn, dev_ArcOut, startA, endA);
					CUDAoutDiff <<<numOfBlocksN, ThxBlock >>> (dev_BPerr, dev_NeuronOut, startN, endN);
					cudaStatus = cudaMemcpy(CBPerr, dev_BPerr, BPerr.size() * sizeof(float), cudaMemcpyDeviceToHost);
					if (cudaCheckStatus(cudaStatus) == true) goto Error;
					copy(CBPerr, CBPerr + BPerr.size(), BPerr.begin());
				}

				//applico a ogni peso la sua correzione

				startN = NeurInLyr[1] + 1; // la correzione dei bais va applicata dal primo layer nascosto in poi
				endN = NeurInLyr[NeurInLyr.size() - 1];

				numLayerNeur = endN - startN + 1;

				numOfBlocksA = floorf(weights.size() / ThxBlock) + 1;
				numOfBlocksN = floorf(numLayerNeur / ThxBlock) + 1;

				CUDAapplyWeightCorrections << <numOfBlocksA, ThxBlock >> > (eps, dev_NeuronOut, dev_BPerr, dev_weights, dev_ArcIn, dev_ArcOut, weights.size());
				CUDAapplyBayesCorrections << <numOfBlocksN, ThxBlock >> > (eps, dev_BPerr, dev_Bayes, startN, endN);

				////////////////////DEBUG SECTION////////////////////////
				//cudaStatus = cudaMemcpy(Cweights, dev_weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
				//if (cudaCheckStatus(cudaStatus) == true) goto Error;
				//copy(Cweights, Cweights + weights.size(), weights.begin());

				//cudaStatus = cudaMemcpy(CBayes, dev_Bayes, Bayes.size() * sizeof(float), cudaMemcpyDeviceToHost);
				//if (cudaCheckStatus(cudaStatus) == true) goto Error;
				//copy(CBayes, CBayes + Bayes.size(), Bayes.begin());

				//cudaStatus = cudaMemcpy(CBPerr, dev_BPerr, BPerr.size() * sizeof(float), cudaMemcpyDeviceToHost);
				//if (cudaCheckStatus(cudaStatus) == true) goto Error;
				//copy(CBPerr, CBPerr + BPerr.size(), BPerr.begin());
				/////////////////////////////////////////////////////////

			}

			t1 = PerformanceCounter();
			elapsedMilliseconds = ((t1 - t0) * 1000.0) / PerformanceFrequency(); // calcolo il tempo di esecuzione di una iterazione di addestramento (tutto il set)
			MeanErr = MeanErr / (examples.size() / (inputN + outputN)); //calcolo l'errore percentuale medio sul dataset
			elapsedInMilliseconds = elapsedInMilliseconds / (examples.size() / (inputN + outputN));
			cout << "Iterazione: " << it << "  " << MeanErr << " %Err  " << "execution time:" << elapsedMilliseconds << "ms" << endl;
			cout << "mean InputTime: " << elapsedInMilliseconds << "ms" << endl;
			printNetSpecs();
			MeanErr = 0;
		}

		cudaStatus = cudaMemcpy(Cweights, dev_weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		copy(Cweights, Cweights + weights.size(), weights.begin());

		cudaStatus = cudaMemcpy(CBayes, dev_Bayes, Bayes.size() * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		copy(CBayes, CBayes + Bayes.size(), Bayes.begin());
		//checkpoint di errore (se la GPU richiama un qualunque errore ripare da qui)
	Error:

		//libero la memoria nella scheda grafica
		cudaFree(dev_weights);
		cudaFree(dev_ArcIn);
		cudaFree(dev_ArcOut);
		cudaFree(dev_NeuronOut);
		cudaFree(dev_examples);
		cudaFree(dev_BPerr);
		cudaFree(dev_mapMaxOut);
		cudaFree(dev_mapMinOut);
		cudaFree(dev_priority);
		cudaFree(dev_NeurInLyr);

		//ritorno lo stato della GPU
		return cudaStatus;
	}

	//esegue il caricamento nella gpu dei parametri della rete
	cudaError_t hostCUDAuploadNetParams() {

		cudaError_t cudaStatus;

		cudaStatus = cudaSetDevice(GpuID);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;

		//host variables
		float *Cweights = &weights[0];
		int *CArcIn = &ArcIn[0];
		int *CArcOut = &ArcOut[0];
		float *CNeuronOut = &NeuronOut[0];
		float *CBayes = &Bayes[0];
		float *CBPerr = &BPerr[0];
		float *CmapMaxOut = &mapMaxOut[0];
		float *CmapMinOut = &mapMinOut[0];
		float *Cexamples = &examples[0];
		int *CNeurInLyr = &NeurInLyr[0];
		int *Cpriority = &priority[0];
		float *CMeanErr = &MeanErr;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(GpuID);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;

		// Allocate GPU buffers for vectors    
		cudaStatus = cudaMalloc((void**)&gpuNetParams.weights, weights.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.ArcIn, ArcIn.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.ArcOut, ArcOut.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.NeuronOut, NeuronOut.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.Bayes, Bayes.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.BPerr, BPerr.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.mapMaxOut, mapMaxOut.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.mapMinOut, mapMinOut.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.examples, examples.size() * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.NeurInLyr, NeurInLyr.size() * sizeof(int));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.priority, priority.size() * sizeof(int));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.InputRT, inputN * sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMalloc((void**)&gpuNetParams.MeanErr, sizeof(float));
		if (cudaCheckStatus(cudaStatus) == true) goto Error;


		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(gpuNetParams.weights, Cweights, weights.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.ArcIn, CArcIn, ArcIn.size() * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.ArcOut, CArcOut, ArcOut.size() * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.NeuronOut, CNeuronOut, NeuronOut.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.Bayes, CBayes, Bayes.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.BPerr, CBPerr, BPerr.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.mapMaxOut, CmapMaxOut, mapMaxOut.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.mapMinOut, CmapMinOut, mapMinOut.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.examples, Cexamples, examples.size() * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.NeurInLyr, CNeurInLyr, NeurInLyr.size() * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.priority, Cpriority, priority.size() * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
		cudaStatus = cudaMemcpy(gpuNetParams.MeanErr, CMeanErr, sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;

		if (false) {
			Error:
			//libero la memoria nella scheda grafica
			cudaFree(gpuNetParams.weights);
			cudaFree(gpuNetParams.ArcIn);
			cudaFree(gpuNetParams.ArcOut);
			cudaFree(gpuNetParams.NeuronOut);
			cudaFree(gpuNetParams.examples);
			cudaFree(gpuNetParams.BPerr);
			cudaFree(gpuNetParams.mapMaxOut);
			cudaFree(gpuNetParams.mapMinOut);
			cudaFree(gpuNetParams.priority);
			cudaFree(gpuNetParams.NeurInLyr);
			cout << "ERRORE: libero la memoria della gpu. " << endl;
		}
			
		return cudaStatus;
	}

	//esegue il download dalla gpu dei parametri della rete
	cudaError_t hostCUDAdownloadNetParams() {

		cout << "downloading net params from gpu.." << endl;

		float *Cweights = &weights[0];
		float *CBayes = &Bayes[0];

		cudaError_t cudaStatus;

		cudaStatus = cudaSetDevice(GpuID);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;

		cudaStatus = cudaMemcpy(Cweights, gpuNetParams.weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;

		cudaStatus = cudaMemcpy(CBayes, gpuNetParams.Bayes, Bayes.size() * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;


		if (false) {
		Error:
			//libero la memoria nella scheda grafica
			cudaFree(gpuNetParams.weights);
			cudaFree(gpuNetParams.ArcIn);
			cudaFree(gpuNetParams.ArcOut);
			cudaFree(gpuNetParams.NeuronOut);
			cudaFree(gpuNetParams.examples);
			cudaFree(gpuNetParams.BPerr);
			cudaFree(gpuNetParams.mapMaxOut);
			cudaFree(gpuNetParams.mapMinOut);
			cudaFree(gpuNetParams.priority);
			cudaFree(gpuNetParams.NeurInLyr);
			cout << "ERRORE: libero la memoria della gpu. " << endl;
		}

		return cudaStatus;
	}

	//esegue l'input della rete gia addestrata prendendo in input l'esempio dato 
	cudaError_t hostCUDAInputNet(float *input, int ThxBlock) {
		//inportante verificare che l'input abbia la stessa dimansione dell'input della rete

		cudaError cudaStatus;

		cudaStatus = cudaSetDevice(GpuID);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;

		//////////////////lancio dei kernel all'interno della gpu////////////////
		//int ThxBlock = 1024;
		int startA = 0;
		int endA = 0;
		int startN = 0;
		int endN = 0;
		int numLayerArcs = 0;
		int numLayerNeur = 0;
		int numOfBlocksMax = 0;
		int numOfBlocksA = 0;
		int numOfBlocksN = 0;
		int numOfBlocksOut = floorf(outputN / ThxBlock) + 1;
		int outputRef = NeuronOut.size() - outputN;
		long long t0in = 0, t1in = 0;
		double elapsedInMilliseconds = 0;
			
		t0in = PerformanceCounter();
		//resetto il vettore contenente lo stato di attivazione dei neuroni
		numOfBlocksA = (floorf(NeuronOut.size() / ThxBlock) + 1);
		CUDAresetVector <<<numOfBlocksA, ThxBlock >>> (gpuNetParams.NeuronOut, NeuronOut.size());

		cudaStatus = cudaMemcpy(gpuNetParams.InputRT, input, inputN * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaCheckStatus(cudaStatus) == true) goto Error;

		//imposto i valori di input ai neuroni dello strato input
		numOfBlocksA = (floorf(inputN / ThxBlock) + 1);
		CUDAsetSingleInput <<<numOfBlocksA, ThxBlock>>> (gpuNetParams.NeuronOut, inputN, gpuNetParams.InputRT);
		
		//propagazione dell'input nella rete

		startA = 0; // indice di partenza dei vettori archi
		endA = 0; // ultimo indice dei vettori archi
		startN = 0; // indice di partenza dei vettori neuroni
		endN = 0; // ultimo indice dei vettori neuroni

		for (int i = 0; i < priority.size() - 1; i++) { //NB non viene applicata la sigmoide allo strato di input eventulmente correggi

			startA = priority[i] + 1;
			endA = priority[i + 1];

			if (i < priority.size() - 2) {
				startN = NeurInLyr[i + 1] + 1;
				endN = NeurInLyr[i + 2];
			}

			numLayerArcs = endA - startA + 1;
			numLayerNeur = endN - startN + 1;

			numOfBlocksA = floorf(numLayerArcs / ThxBlock) + 1;
			numOfBlocksN = floorf(numLayerNeur / ThxBlock) + 1;

			if (i < priority.size() - 2) {
				CUDAlayerInput <<<numOfBlocksA, ThxBlock >>> (gpuNetParams.weights, gpuNetParams.ArcIn, gpuNetParams.ArcOut, gpuNetParams.NeuronOut, startA, endA); //propago l'output dei neuroni al prossimo/i layer
				CUDAbayesInput <<<numOfBlocksN, ThxBlock >>> (gpuNetParams.NeuronOut, gpuNetParams.Bayes, startN, endN); //applico il contributo dei bayes all output dei neuroni del layer corrente 
				CUDAsigLayer <<<numOfBlocksN, ThxBlock >>> (gpuNetParams.NeuronOut, startN, endN); //applico la sigmoide allo stato di attivazione dei neuroni
				
			}
		}

		//copio l'output dei neuroni dello strato output nella memoria della cpu
		cudaStatus = cudaMemcpy(&NeuronOut[0] + outputRef, gpuNetParams.NeuronOut + outputRef, outputN * sizeof(float), cudaMemcpyDeviceToHost); //TODO da errore e non carica il vettore trovare il BUG
		if (cudaCheckStatus(cudaStatus) == true) goto Error;
			
		t1in = PerformanceCounter();
		elapsedInMilliseconds = ((t1in - t0in) * 1000.0) / PerformanceFrequency();

		////////////////////////////////////visualizzazione dell'esempio///////////////////////////////////

		float delta;
		cout << "input time: " << elapsedInMilliseconds << " ms" << endl;

		for (int on = 0; on < outputN; on++) {
			delta = mapMaxOut[on] - mapMinOut[on];
			cout << "Y" << on << ": " << (NeuronOut[NeuronOut.size() - outputN + on] * delta) + mapMinOut[on] << endl;
		}
		cout << endl;
		
		///////////////////////////////////////////////////////////////////////////////////////////////////

		if (false) {
		Error:
			//libero la memoria nella scheda grafica
			cudaFree(gpuNetParams.weights);
			cudaFree(gpuNetParams.ArcIn);
			cudaFree(gpuNetParams.ArcOut);
			cudaFree(gpuNetParams.NeuronOut);
			cudaFree(gpuNetParams.examples);
			cudaFree(gpuNetParams.BPerr);
			cudaFree(gpuNetParams.mapMaxOut);
			cudaFree(gpuNetParams.mapMinOut);
			cudaFree(gpuNetParams.priority);
			cudaFree(gpuNetParams.NeurInLyr);
		}

		return cudaStatus;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////CUDA UTILITY//////////////////////////////////////////////////////////////////
	//verifica la corretta esecuzione di un operazione
	inline cudaError_t checkCuda(cudaError_t result)
	{
#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			assert(result == cudaSuccess);
		}
#endif
		return result;
	}
	//verifica la corretta esecuzione di un operazione restituendo un bool
	bool cudaCheckStatus(cudaError_t cudaStatus) {
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			return true;
		}
	}
	//stampa a schermo le principali proprietà della scheda
	void printDeviceSpecs() {

		printf("\nDevice: %s\n", prop.name);
		printf("Cores clock: %d MHz\n", (prop.clockRate / 1000));
		printf("Memory clock: %d MHz\n", (prop.memoryClockRate / 1000));
		printf("Total global memmory %.2f MB\n", (float)(prop.totalGlobalMem / (1024 * 1024)));
		printf("Max grid size: x %d, y %d, z %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Max block axis size: x %d, y %d, z %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Warp size: %d\n", prop.warpSize);
		printf("Max therads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max therads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
		printf("Compute Mode: %d\n", prop.computeMode);
		printf("Host mem access: %d\n", prop.canUseHostPointerForRegisteredMem);
		printf("Shared mem per multiprocessor %.2f KB\n", (float)(prop.sharedMemPerMultiprocessor / 1024));
		printf("Max shared mem per blocco %.2f KB\n", (float)(prop.sharedMemPerBlock / 1024));

	}
	//stampa i parametri della rete che vengono passati alla scheda
	void printNetSpecs() {
		cout << "dimensione del modello: " << sizeOfModel("MB") << " MB" << endl;
		cout << "numero totale dei neuroni : " << NeuronOut.size() << "(" << sizeOfVector(NeuronOut, "KB") + sizeOfVector(Bayes, "KB") + sizeOfVector(BPerr, "KB")<< " KB)" << endl;
		cout << "numero totale degli archi : " << weights.size() << "(" << sizeOfVector(weights, "MB") + sizeOfVector(ArcIn, "MB") + sizeOfVector(ArcOut, "MB") << " MB)" << endl;
		cout << "numero esempi : " << examples.size() / (inputN + outputN) << "  (" << sizeOfVector(examples, "MB") << " MB)" << endl;
	}
	//calcola il peso del modello
	float sizeOfModel(string mesureUnit = "B") {
		int size = 0;
		int scale = 0;
		size += sizeof(weights[0])*weights.size(); //dimensione del vettore pesi
		size += sizeof(ArcIn[0])*ArcIn.size(); //dimensione del vettore contenente i target degli archi
		size += sizeof(ArcOut[0])*ArcOut.size(); //dimensione del vettore contenete i neuroni base degli archi
		size += sizeof(NeuronOut[0])*NeuronOut.size(); //dimensione del vettore contenente gli output dei neuroni
		size += sizeof(Bayes[0])*Bayes.size(); //dimensione del vettore contenente i bayes
		size += sizeof(BPerr[0])*BPerr.size(); //dimensione del vettore contenente 
		size += sizeof(priority[0])*priority.size(); // dimensione del vettore priorità
		size += sizeof(examples[0])*examples.size(); //dimensione del vettore di esmpio
		if (mesureUnit == "B") { scale = 1; } //Byte
		else if (mesureUnit == "KB") { scale = 1024; } //Kilobyte
		else if (mesureUnit == "MB") { scale = 1024 * 1024; } // Megabyte
		else if (mesureUnit == "GB") { scale = 1024 * 1024 * 1024; } //Gigabyte
		else { cout << "L'unità di misura non è corretta!!" << endl; return 0.0f; }
		return (float)(size / scale);
	}
	template<typename T, typename A>
	float sizeOfVector(vector<T, A> const& vect,string mesureUnit = "B") {
		int size = 0;
		int scale = 0;
		size = sizeof(vect[0])*vect.size(); //dimensione del vettore di esmpio
		if (mesureUnit == "B") { scale = 1; } //Byte
		else if (mesureUnit == "KB") { scale = 1024; } //Kilobyte
		else if (mesureUnit == "MB") { scale = 1024 * 1024; } // Megabyte
		else if (mesureUnit == "GB") { scale = 1024 * 1024 * 1024; } //Gigabyte
		else { cout << "L'unità di misura non è corretta!!" << endl; return 0.0f; }
		return (float)((float)size /(float)scale);
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////ALTRE FUNZIONI////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////WINDOWS HIGH SPEED TIMING//////////////////////////////////////////////////////
	BOOL WINAPI QueryPerformanceCounter(_Out_ LARGE_INTEGER *lpPerformanceCount);
	BOOL WINAPI QueryPerformanceFrequency(_Out_ LARGE_INTEGER *lpFrequency);
	inline long long PerformanceCounter() noexcept
	{
		LARGE_INTEGER li;
		::QueryPerformanceCounter(&li);
		return li.QuadPart;
	}
	inline long long PerformanceFrequency() noexcept
	{
		LARGE_INTEGER li;
		::QueryPerformanceFrequency(&li);
		return li.QuadPart;
	}
	/* HOW TO USE:
	long long t0 = PerformanceCounter();
	//code to bench..
	long long t1 = PerformanceCounter();
	double elapsedMilliseconds = ((t1 - t0) * 1000.0) / PerformanceFrequency();
	*/
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};


class genEvolve {
private:
	vector<MLP> mlps; // vettore di reti mlp
	vector<Hopfield> hpds; // vettore di reti Hopfield
public:

};


int main()
{
	CUDAcore gtx760(0);
	gtx760.printDeviceSpecs();

	int layer = 10;
	int columns = 50;
	int out = 4;
	int in = 9;

	MLP a("cudaTest");
	//a.getNetParams();
	a.qubeNet(layer, columns, in, out, true, 0.0001f);
	a.setNetMap(800, 0);
	//a.genTestDataset(500, in, out, 0.1, 3, 0);
	a.getDataset("datnc");
	a.datasetOffset(200.0f);
	//float test[] = {2, 2, 2, 2, 2, 2};
	//a.BP(5, 0.0000000001f, 0, 1);
	gtx760.cudaNetCopyMLP(&a);
	gtx760.cudaNetCopyExamples(&a);
	
	//da usare senza " gtx760.hostCUDAtrainingNet "
	//gtx760.hostCUDAuploadNetParams();
	//gtx760.hostCUDAInputNet(test, 1024);
	//gtx760.hostCUDAdownloadNetParams();

	gtx760.hostCUDAtrainingNet(10.0e-10f, 200, 256);
	gtx760.cudaNetPasteMLP(&a);

	a.saveNet("datncNet-l(5)-c(60)-FC");

	a.BP(5, 10.0e-10f, 0, 1);

	a.saveNet("CpuTest5");
	/*
	system("pause");
	int layer = 3;
	int columns = 50;
	int out = 7;
	int in = 7;
	//MLP a("GenomaX2");
	MLP b("GenomaX2");
	//b.getNetParams();
	b.qubeNetFC(layer, columns, in, out, false);

	//DatasetCore data;
	//data.readTimeSeriesCsv("LottoHistory", out, in/out, 100);
	//b.examples = data.getDataset(0);
	//b.saveDataset("LotoDatatset");

	//b.getDataset("LotoDatatset");
	//vector<int> dims = {in, columns, out};
	//b.customNet(layer, dims, 1);

	Hopfield a("genoma4", &b);
	StructuralLearning SL(&b, &a);

	//a.saveNet("GenomaX");
	////a.toroidNet(layer, dims, 0.3);
	////a.supportNet(1);
	b.setNetMap(4, 0);

	//cout << "caricamento eseguito" << endl;
	b.genTestDataset(50, in, out, 0.2, 3, 2);
	b.BP(200, 0.0001, 0.2, 5);
	//SL.StructuralBP(200, 0.2, 0.3, 0.4, 0.001, -0.001, 8, 5, 0.05, 3, 40);
	////a.getDataset("Dataset");
	////cout << "dataset genrato" << endl;

	//b.BP(200, 0.2, 0.3, 0.9);
	////cout << "addestramento completato" << endl;
	//a.saveNet("GenomaX5");
	//b.saveNet("GenomaX");
	b.saveNet("GenomaX2");
	//cout << "salvataggio eseguito" << endl;
	////system("pause");
	*/
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*NOTE:
	_______________________________________________________________________________________________________________________________
	TASKS COMPLETATE:

	-Hopfield::suportnet() ora crea una struttura su una sola linea con le connessioni nello stesso ordine e verso con i rispettvi
		neuroni delle due reti

	-fuzioni aggiornate per il cambio di puntamento dei vettori di propagazione:
		stampInputInfluences(), stampOutputErrorPropagation(), initVectorsProfiler(), resetVectorsProfiler(), StructuralBP(), inputNetProfiler()
	_______________________________________________________________________________________________________________________________

	TASKS TODO:

	-creare delle funzioni di eliminazione arco e neurone apposite per la struttura mlp + supportNet + binds nella classe Hopfield
		e rimettere aposto le funzioni di eliminazione arco e neurone nella classe network

	-rivedere la funzione NetInputProfiler() nella procedura per il calcolo dell' influnza dell'input (procedura inefficiente)

	-determinare una funzione che possa essere utilizzata anche per Hopfield::AssociativeCorrelation()

	-scrivere la funzione di apprendimento della rete hopfield di supporto

	-aggiungere la propagazione temporle pesata dell'output del neurone

	-verificare che non ci siano bug nella funzione trainSupportNet() (possibile scambio di indici)

	-nella funzione taglia archi si possono verificare dei problemi eliminando degli archi che eliminano neuroni a catena
		il problema è stato risolto per il taglio indiretto di un neurone ma non per piu consecutivi per i quali non viene
		eliminato il bind e probabilmente alcuni parametri non vengono modificati

	-La funzione hostCUDAtrainingNet() è affetta da RACE CONDITIONS rivedere l'algoritmo per minimizare l'utilizzo
		delle Atomic functions .. la correzione va estesa alle funzioni che costruiscono la struttura linearizata della rete da
		passare alla GPU
	________________________________________________________________________________________________________________________________

	POSSIBILI PATCH:

	*/