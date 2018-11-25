#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////

int L = 3;

int numTrained = 60;

int numInput = 150;

int N[]={4,5,3};

double w[20][1000][1000];

double b[20][1000];

double a[20][1000][150];

double Z[20][1000][150];

double type[150][3];

string predict[150];

double S(double i){
    double out;
    out=1/(1+exp(-i));
    return out;
};

void setWeight(int l, int n, int p, double weight){
    w[l][n][p]=weight;
}

void setBias(int l, int n, double bias){
    b[l][n]=bias;
}

void activationLoop(int i){
    for (int l=1; l<L; ++l) {
        for (int n = 0; n < N[l]; ++n) {
            Z[l][n][i] = 0;
            for (int p = 0; p < N[l - 1]; ++p) {
                Z[l][n][i] = Z[l][n][i] + (w[l][n][p] * a[l - 1][p][i]);
            }
            Z[l][n][i] = Z[l][n][i] + b[l][n];
            a[l][n][i] = S(Z[l][n][i]);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


void getInput(){

    ifstream inputFile;
    inputFile.open(R"(C:\Users\User\CLionProjects\Iris-Flower\Iris.txt)");

    while (inputFile.good()and !inputFile.eof()){
        for (int i=0; i<150; ++i) {
            for (int n = 0; n < N[0]; ++n) {
                string in;
                getline(inputFile, in, ',');
                a[0][n][i]=stod(in);
            }
            string ft;
            getline(inputFile, ft, '\n');

            if (ft=="setosa"){
                type[i][0]=1;
                type[i][1]=0;
                type[i][2]=0;
            }else if (ft=="versicolor"){
                type[i][0]=0;
                type[i][1]=1;
                type[i][2]=0;
            }else if (ft=="virginica"){
                type[i][0]=0;
                type[i][1]=0;
                type[i][2]=1;
            }
        }
    }

    inputFile.close();
};

double getError(int n, int i){
    double err;
    err=pow((type[i][n]-a[L-1][n][i]),2);
    return err;
}


double avgError(){
    double errSum=0;
    double avErr;
    int runs=0;
    for (int i = 0; i < numTrained; ++i) {
        for (int n = 0; n < N[L - 1]; ++n) {
            errSum = errSum + getError(n, i);
            runs=runs+1;
        }
    }
    avErr=errSum/runs;
    return avErr;
}

void train() {

    for(int i=0; i<numTrained; ++i){
        activationLoop(i);
    }

    int layer = 1 + rand() % (L - 1);
    int neuron = rand() % (N[layer]);
    int prevNeur = rand() % N[layer - 1];
    double change = (((double(rand()) / RAND_MAX) * 2.0) - 1.0) * 0.1;

    double prevWeight = w[layer][neuron][prevNeur];
    double prevBias = b[layer][neuron];

    double errBef=avgError();

    double biasOrWeight = double(rand()) / RAND_MAX;

    if (biasOrWeight<=0.8){
        setWeight(layer, neuron, prevNeur, prevWeight + change);
    }else if (biasOrWeight>0.8){
        setBias(layer, neuron, prevBias + change);
    }

    for(int i=0;i<numTrained;++i){
        activationLoop(i);
    }

    double errAft=avgError();

    if(errBef<=errAft){
        setWeight(layer, neuron, prevNeur, prevWeight);
        setBias(layer,neuron,prevBias);
        //cout<<"Unchanged"<<endl;
    }else{
        //cout<<"Changed"<<endl;
    }
}

void setPredict(){
    ofstream output;
    output.open(R"(C:\Users\User\CLionProjects\Iris-Flower\prediction.txt)");

    for(int i=0;i<numInput;++i){

        if(a[2][0][i] > a[2][1][i] && a[2][0][i] > a[2][2][i]){
            predict[i]="Setosa";
        }else if(a[2][1][i] > a[2][0][i] && a[2][1][i] > a[2][2][i]){
            predict[i]="Versicolor";
        }else if(a[2][2][i] > a[2][0][i] && a[2][2][i] > a[2][1][i]){
            predict[i]="Virginica";
        }

        output<<i+1<<": "<<predict[i]<<endl;

    }
    output.close();

}

void saveBias(){
    ofstream bias;
    bias.open(R"(C:\Users\User\CLionProjects\Iris-Flower\bias.txt)");

    for(int l=1; l<L;++l){
        for(int n=0; n<N[l]; ++n){
            bias<<b[l][n];
            if(n!=N[l]-1){
                bias<<",";
            }
        }
        if(l!=L-1){
            bias<<endl;
        }
    }
    bias.close();

}

void saveWeights(){
    ofstream weights;
    weights.open(R"(C:\Users\User\CLionProjects\Iris-Flower\weights.txt)");

    for (int l=1; l<L; ++l) {
        for (int n = 0; n < N[l]; ++n) {
            for (int p = 0; p < N[l - 1]; ++p) {
                weights<<w[l][n][p];
                if (p!=N[l-1]-1){
                    weights<<",";
                }
            }
            if(!(n==N[l]-1 & l==L-1)){
                weights<<endl;
            }
        }
    }
    weights.close();

}

void readBias(){
    ifstream biasIn;
    biasIn.open(R"(C:\Users\User\CLionProjects\Iris-Flower\bias.txt)");

    while (biasIn.good()and !biasIn.eof()){
        for(int l=1; l<L;++l){
            for(int n=0; n<N[l]; ++n) {
                string inB;
                if (n!=N[l]-1){
                    getline(biasIn, inB, ',');
                    b[l][n]=stod(inB);
                } else{
                    getline(biasIn, inB, '\n');
                    b[l][n]=stod(inB);
                }
            }
        }
    }
    biasIn.close();
};

void readWeights(){
    ifstream weightsIn;
    weightsIn.open(R"(C:\Users\User\CLionProjects\Iris-Flower\weights.txt)");

    while (weightsIn.good()and !weightsIn.eof()){
        for (int l=1; l<L; ++l) {
            for (int n = 0; n < N[l]; ++n) {
                for (int p = 0; p < N[l - 1]; ++p) {
                    string inW;
                    if (p!=N[l-1]-1){
                        getline(weightsIn, inW, ',');
                        w[l][n][p]=stod(inW);
                    } else{
                        getline(weightsIn, inW, '\n');
                        w[l][n][p]=stod(inW);
                    }
                }
            }
        }
    }
    weightsIn.close();

}

int main(int argc, const char * argv[]) {

    for(int l=0;l<20;++l){
        for(int n=0;n<1000;++n){
            for(int p=0;p<1000;++p){
                double setW=((double(rand())/RAND_MAX)*2.0)-1.0;
                setWeight(l,n,p,setW);
            }
            double setB=((double(rand())/RAND_MAX)*2.0)-1.0;
            setBias(l,n,setB);
        }
    };

    getInput();

    for(int i=0;i<100000;++i){
        train();
    }

    for(int i=0;i<numInput; ++i){
        activationLoop(i);
    }

    setPredict();
    saveWeights();
    saveBias();

    readBias();
    readWeights();

    for(int i=0;i<numInput; ++i){
        activationLoop(i);
    }

    return 0;
}