#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////

int L = 4;

int numTrained = 500;

int numInput = 5000;

double cutoff = 0.2;

int N[]={20,10,5,2};

double w[20][1000][1000];

double b[20][1000];

double a[20][1000][5000];

double Z[20][1000][5000];

double depth[5000];

double duration[5000];

double location[5000];

double predict[5000];

double exo[5000][2];

double normalDepth[5000];

int falPos = 0;

int falNeg = 0;

int truPos = 0;

int truNeg = 0;

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
    inputFile.open(R"(C:\Users\User\CLionProjects\Exoplanet\NormalSample.txt)");

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

            string nrm;
            getline(inputFile, nrm, '\r');
            normalDepth[i]=stod(nrm);

            if(depth[i] == 0){
                exo[i][0]=1;
                exo[i][1]=0;
            }else{
                exo[i][0]=0;
                exo[i][1]=1;
            }

        }
    }

    inputFile.close();
};

double getError(int n, int i){
    double err;
    err=pow((exo[i][n]-a[L-1][n][i]),2);
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

void train(double multiplier) {

    int layer = 1 + rand() % (L - 1);
    int neuron = rand() % (N[layer]);
    int prevNeur = rand() % N[layer - 1];
    double change = (((double(rand()) / RAND_MAX) * 2.0) - 1.0) * multiplier;

    double prevWeight = w[layer][neuron][prevNeur];
    double prevBias = b[layer][neuron];

    for(int i=0;i<numTrained; ++i){
        activationLoop(i);
    }

    double errBef=avgError();

    double biasOrWeight = double(rand()) / RAND_MAX;

    if (biasOrWeight<=0.8){
        setWeight(layer, neuron, prevNeur, prevWeight + change);
    }else if (biasOrWeight>0.8){
        setBias(layer, neuron, prevBias + change);
    }

    for(int i=0;i<numTrained; ++i){
        activationLoop(i);
    }

    double errAft=avgError();

    //cout<<"Before: "<<errBef<<", After: "<<errAft<<endl;

    if(errBef<=errAft){
        setWeight(layer, neuron, prevNeur, prevWeight);
        setBias(layer, neuron, prevBias);
        for(int i=0;i<numTrained; ++i){
            activationLoop(i);
        }
        //cout<<"Unchanged"<<endl;
    }else{
        //cout<<"Changed"<<endl;
    }
}

void setPredict(){
    ofstream output;
    output.open(R"(C:\Users\User\CLionProjects\Exoplanet\prediction.txt)");

    output<<"Index: Actual, Normalised, Neuron 1, Neuron 2, Prediction"<<endl;

    for(int i=0;i<numInput;++i){

        if (a[L-1][0][i]>a[L-1][1][i]){
            predict[i]=0;
        }else{
            predict[i]=1;
        }

        output<<i+1<<": "<<depth[i]<<", "<<normalDepth[i]<<", "<<a[L-1][0][i]<<", "<<a[L-1][1][i]<<", "<<predict[i]<<endl;

    }

    output<<"True Positives: "<<truPos<<endl<<"False Positives: "<<falPos<<endl<<"True Negatives: "<<truNeg<<endl<<"False Negatives: "<<falNeg<<endl;

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

void positiveTest(){

    for (int i=0; i<numInput; ++i){
        if (a[L-1][0][i]>a[L-1][1][i] && exo[i][0]==1){
            truNeg = truNeg + 1;
        }else if (a[L-1][0][i]>a[L-1][1][i] && exo[i][0]==0){
            falNeg = falNeg + 1;
        }else if (a[L-1][0][i]<a[L-1][1][i] && exo[i][1]==1){
            truPos = truPos + 1;
        }else if (a[L-1][0][i]<a[L-1][1][i] && exo[i][1]==0){
            falPos = falPos + 1;
        }else{
            cout<<i+1<<": Equal!"<<endl;
            truPos = truPos + 1;
        }
    }

}

int main(int argc, const char * argv[]) {

    randBandW();

    getInput();

    /*readBias();
    readWeights();*/

    for(int i=0;i<numInput; ++i){
        activationLoop(i);
    }

    for(int i=0;i<1000000;++i){
        if(i % 10000==0){
            cout<<i<<": "<<avgError()<<endl;
        }

        if(i>0 && i<200000){
            train(100);
        }else if(i>=200000 && i<400000){
            train(10);
        }else if(i>=400000 && i<600000) {
            train(1);
        }else if(i>=600000 && i<800000) {
            train(0.1);
        }else if(i>=800000 && i<1000000) {
            train(0.01);
        }

    }

    for(int i=0;i<numInput; ++i){
        activationLoop(i);
    }

    positiveTest();

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