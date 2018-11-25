#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////

int L = 5;

int numTrained = 100;

int numInput = 5000;

int N[]={20,10,5,3,1};

double w[20][1000][1000];

double b[20][1000];

double a[20][1000][5000];

double Z[20][1000][5000];

double depth[5000];

double duration[5000];

double location[5000];

double predict[5000];

double exo[5000];

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
    inputFile.open(R"(C:\Users\User\CLionProjects\Exoplanet\sample.txt)");

    while (inputFile.good()and !inputFile.eof()){

        for (int i=0; i<numInput; ++i) {
            for (int n = 0; n < N[0]; ++n) {
                string in;
                getline(inputFile, in, ',');
                a[0][n][i]=stod(in);
            }
            string dep;
            getline(inputFile, dep, ',');
            depth[i]=stod(dep);
            string dur;
            getline(inputFile, dur, ',');
            duration[i]=stod(dur);
            string loc;
            getline(inputFile, loc, ',');
            location[i]=stod(loc);

            if(depth[i] > 0){
                exo[i]=1;
            }else{
                exo[i]=0;
            }

        }
    }

    inputFile.close();
};

double getError(int n, int i){
    double err;
    err=pow((exo[i]-a[L-1][n][i]),2);
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
    double change = (((double(rand()) / RAND_MAX) * 2.0) - 1.0) * 1;

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
        setBias(layer, neuron, prevBias);
        //cout<<"Unchanged"<<endl;
    }else{
        //cout<<"Changed"<<endl;
    }
}

void setPredict(){
    ofstream output;
    output.open(R"(C:\Users\User\CLionProjects\Exoplanet\prediction.txt)");

    output<<"Index: Actual, Prediction"<<endl;

    for(int i=0;i<numInput;++i){

        predict[i]=a[L-1][0][i];

        output<<i+1<<": "<<depth[i]<<", "<<predict[i]<<endl;

    }
    output.close();

}

void saveBias(){
    ofstream bias;
    bias.open(R"(C:\Users\User\CLionProjects\Exoplanet\bias.txt)");

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
    weights.open(R"(C:\Users\User\CLionProjects\Exoplanet\weights.txt)");

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
    biasIn.open(R"(C:\Users\User\CLionProjects\Exoplanet\bias.txt)");

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
    weightsIn.open(R"(C:\Users\User\CLionProjects\Exoplanet\weights.txt)");

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

void randBandW(){
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
}

int main(int argc, const char * argv[]) {

    randBandW();

    getInput();

    for(int i=0;i<10000000;++i){
        train();
        if(i % 1000==0){
            //cout<<i<<": "<<avgError()<<endl;
        }
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