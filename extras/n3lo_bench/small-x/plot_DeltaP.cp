/* -----------------------------------------

   Test of HELL - plots

   ----------------------------------------- */
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <sys/time.h>
#include <getopt.h>
//#include <gsl/gsl_sf.h>

#include "hell.hh"
#include "hell-x.hh"
#include "Pqg.hh"
#include "gammaNLO.hh"
#include "gammaNNLO.hh"
#include "math/special_functions.hh"


using namespace std;


int _damping = 2, _dampingsqrt = 4;


const double Nc = 3.;
const double CA = Nc;
const double CF = (Nc*Nc-1.)/2./Nc;
const double ZETA2  =  1.64493406684822643647;
//const double ZETA3  =  1.2020569031595942855;
//const double ZETA4  =  1.082323233711138191516;

double Pqg0(double x, double nf) {
  return nf/2./M_PI*(x*x+(1-x)*(1-x));
}
double Pqg1(double x, double nf) {
  double lnx   = log(x);
  double ln1mx = log(1-x);
  double pqg   = x*x + (1-x)*(1-x);
  double pqgmx = x*x + (1+x)*(1+x);
  double S2x   = -2*HELLx::Li2(-x) + lnx*lnx/2. - 2*lnx*log(1+x) - ZETA2;
  //
  double X1QGA =
    2*CF*( 4+4*ln1mx + (10-4*(ln1mx-lnx) + 2*pow(-ln1mx+lnx,2) - 4*ZETA2) * pqg
	   - lnx * (1-4*x) - lnx*lnx * (1-2*x) - 9*x )
    + 2*CA* ( 182./9. -4*ln1mx
	      + ( - 218./9. + 4*ln1mx - 2*ln1mx*ln1mx + (44*lnx)/3. - lnx*lnx + 2*ZETA2 ) * pqg
	      + 2*pqgmx*S2x + 40./(9*x) + 14*x/9. - lnx*lnx*(2+8*x)
	      + lnx * (-38./3. + 136.*x/3.) );
  return nf * X1QGA /pow(4*M_PI,2);
}

dcomplex DeltaGamma(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  return Delta_gamma;
}
dcomplex DeltaGammaDot(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  return Delta_gammadot;
}
dcomplex GammaNLOgg(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  N += 1;
  return as*gamma0gg(N, nf) + as*as*gamma1SGgg(N, nf);
}
dcomplex GammaNLOqg(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  N += 1;
  return as*gamma0qg(N, nf) + as*as*gamma1SGqg(N, nf);
}
dcomplex GammaNLOgq(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  N += 1;
  return as*gamma0gq(N) + as*as*gamma1SGgq(N, nf);
}
dcomplex GammaNLOqq(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  N += 1;
  return as*gamma0qq(N) + as*as*gamma1SGqq(N, nf);
}
dcomplex GammaNNLOgg(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  N += 1;
  return as*gamma0gg(N, nf) + as*as*gamma1SGgg(N, nf) + as*as*as*gamma2SGgg(N, nf);
}
dcomplex GammaNNLOqg(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  N += 1;
  return as*gamma0qg(N, nf) + as*as*gamma1SGqg(N, nf) + as*as*as*gamma2SGqg(N, nf);
}
dcomplex GammaNNLOgq(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  N += 1;
  return as*gamma0gq(N) + as*as*gamma1SGgq(N, nf) + as*as*as*gamma2SGgq(N, nf);
}
dcomplex GammaNNLOqq(dcomplex Delta_gamma, dcomplex Delta_gammadot, dcomplex N, double as, double nf, dcomplex Delta_gammaNLO, void *p) {
  N += 1;
  return as*gamma0qq(N) + as*as*gamma1SGqq(N, nf) + as*as*as*gamma2SGqq(N, nf);
}


double PggLO(double x, double as, double nf) {
  return as*3./M_PI*(x/(1-x)+(1-x)/x+x*(1-x));
}
double PgqLO(double x, double as, double nf) {
  return as*2./3./M_PI*(2-2*x+x*x)/x;
}
double PqgLO(double x, double as, double nf) {
  return as*Pqg0(x,nf);
}
double PqqLO(double x, double as, double nf) {
  return as/M_PI * 2./3.*(1+x*x)/(1-x);
}

double PqgNLO(double x, double as, double nf) {
  return as*Pqg0(x,nf) + as*as*Pqg1(x,nf);
}



double ccssX[400], ccss[3][4][400];
void readCCSS() {
  string name1[] = { "../hell/output/CCSS/NxNQ5_Nrun_src1_split_lk1.8_dk0.025.sdat",
		     "../hell/output/CCSS/NxNQ5_Nrun_lnmu0.7_src1_split_lk1.8_dk0.025.sdat",
		     "../hell/output/CCSS/NxNQ5_Nrun_lnmu-0.7_src1_split_lk1.8_dk0.025.sdat" };
  string name2[] = { "../hell/output/CCSS/NxNQ5_Nrun_src2_split_lk1.8_dk0.025.sdat",
		     "../hell/output/CCSS/NxNQ5_Nrun_lnmu0.7_src2_split_lk1.8_dk0.025.sdat",
		     "../hell/output/CCSS/NxNQ5_Nrun_lnmu-0.7_src2_split_lk1.8_dk0.025.sdat" };
  ifstream infile[3][2];
  string dummy;
  for(int i=0; i<3; i++) {
    infile[i][0].open(name1[i]);
    infile[i][1].open(name2[i]);
    for(int k=0; k<400; k++) {
      infile[i][0] >> dummy >> ccssX[k] >> ccss[i][0][k] >> ccss[i][2][k];
      infile[i][1] >> dummy >> dummy    >> ccss[i][1][k] >> ccss[i][3][k];
      //cout << ccssX[k] << endl;
    }
  }
}
double CCSS(double x, string chan, string var) {
  int ch = 0, v = 0;
  if(chan=="gq") ch = 1;
  if(chan=="qg") ch = 2;
  if(chan=="qq") ch = 3;
  if(var=="u") v=1;
  if(var=="d") v=2;
  int i = -1;
  for(int k=0; k<400; k++) {
    if(x>=ccssX[k]) break;
    i++;
  }
  if(i>=399 || i<3) return __builtin_nan("");
  //cout << x << "  " << ccssX[i]  << "  " << ccssX[i+1] << endl;
  //double factor = (x - ccssX[i]) / (ccssX[i+1] - ccssX[i]);
  double factor = (log(x) - log(ccssX[i])) / (log(ccssX[i+1]) - log(ccssX[i]));
  return ccss[v][ch][i] + factor * ( ccss[v][ch][i+1] - ccss[v][ch][i] );
}













string sas(double as) {
  ostringstream os;
  if     (as<0.01) os << "000" << int(1000*as);
  else if(as<0.1 ) os << "00"  << int(1000*as);
  else if(as<1.  ) os << "0"   << int(1000*as);
  else os << int(1000*as);
  return os.str();
}



void print_usage() {
  cout << endl
       << "Usage:" << endl
       << " -a  --alphas <as>      set alpha_s" << endl
       << " -n  --nf <nf>          set the number of active flavours" << endl
       << " -L  --useLLp           use LLp rather than NLL" << endl
       << endl;
  exit(0);
  return;
}

void read_arguments(int argc, char* argv[], double &as, int &nf, bool &useLLp) {
  const char* const short_options = "ha:n:L";
  const struct option long_options[] = { { "help", 0, NULL, 'h' },
					 { "alphas", 1, NULL, 'a' },
		      			 { "nf", 1, NULL, 'n' },
		      			 { "useLLp", 0, NULL, 'L' },
		      			 { NULL, 0, NULL, 0 } };
  int next_option;
  ostringstream sset;
  do {
    next_option = getopt_long (argc, argv, short_options, long_options, NULL);
    switch (next_option) {
    case 'h':
      print_usage();
    case 'a':
      as = strtod(optarg, NULL);
      break;
    case 'n':
      nf = int(strtod(optarg, NULL));
      break;
    case 'L':
      useLLp = true;
      break;
    case '?':
      print_usage();
    case -1: break;
    default: abort();
    }
  }
  while (next_option != -1);
  return;
}



int main (int argc, char* argv[]) {

  struct timeval t0, t1;
  gettimeofday(&t0,NULL);

  int nf = 4;
  double as = 0.2;
  bool useLLp = false;

  read_arguments(argc,argv,as,nf,useLLp);

  cout << "nf = " << nf << endl
       << "as = " << as << endl;

  HELLx::HELLxnf sxDLL (nf, HELLx::LL,  "../hell-x/data/");
  HELLx::HELLxnf sxD   (nf, HELLx::NLL, "../hell-x/data/");

  HELL::HELLnf   sxDv  (nf, HELL::NLL,  "../hell/data/");
  HELL::HELLnf   sxD0  (nf, HELL::NLL,  "../hell/data/");
  HELL::HELLnf   sxD0LL(nf, HELL::LL,   "../hell/data/");

  if(useLLp) {
    sxDLL.SetLLpMode(true);
    sxD  .SetLLpMode(true);
    HELL::SetLLpMode(true);
  }

  int Npoints = 9;
  double x = 1e-9;


  readCCSS();


  Npoints = 1525;
  Npoints = 225;
  double x_min = 1e-9;
  double x_max = 0.97;
  ostringstream filename;
  filename << "output/P_as0" << 100*as << "_nf" << nf << (useLLp ? "_LLp" : "") << ".dat";
  ofstream ofile2(filename.str().c_str());
  //
  HELL::HellTableP htpgg, /*htpqg,*/ htpgq, htpqq;
  HELL::HellTableP htpggNNLO, htpqgNNLO, htpgqNNLO, htpqqNNLO;
  filename.str(""); filename.clear();
  filename << "../hell/data/N-space/gammaLLp_nf" << nf << "_alphas" << sas(as) << ".table";
  htpgg.readTable(filename.str());
  //htpqg.readTable(filename.str());
  htpgq.readTable(filename.str());
  htpqq.readTable(filename.str());
  htpgg.setMellinFunction(GammaNLOgg);
  //htpqg.setMellinFunction(GammaNLOqg);
  htpgq.setMellinFunction(GammaNLOgq);
  htpqq.setMellinFunction(GammaNLOqq);
  //
  htpggNNLO.readTable(filename.str());
  htpqgNNLO.readTable(filename.str());
  htpgqNNLO.readTable(filename.str());
  htpqqNNLO.readTable(filename.str());
  htpggNNLO.setMellinFunction(GammaNNLOgg);
  htpqgNNLO.setMellinFunction(GammaNNLOqg);
  htpgqNNLO.setMellinFunction(GammaNNLOgq);
  htpqqNNLO.setMellinFunction(GammaNNLOqq);
  //
  sxDv  .SetLargexDamping(_damping,_dampingsqrt);
  sxDv  .SetMomentumConservation();
  sxD0  .SetLargexDamping(_damping,_dampingsqrt);
  sxD0  .SetMomentumConservation();
  sxD0LL.SetLargexDamping(_damping,_dampingsqrt);
  sxD0LL.SetMomentumConservation();
  //
  for(int i=0; i<Npoints; i++) {
    x = x_min * exp(i/(Npoints-1.)*log(x_max/x_min));
    //
    //HELL::SetRCmodeCF(0);  // default
    double logR0 = sxD0.deltaLogR(as, x);
    sxD0.SetRCvariation(true); // gamma+ variation
    HELL::sqmatrix<double>   d2Pv1 = sxD0.DeltaP(as, x);
    HELLx::sqmatrix<double> xd2Pv1 = HELLx::sqmatrix<double>(d2Pv1.gg(),d2Pv1.gq(),d2Pv1.qg(),d2Pv1.qq());
    sxD0LL.SetRCvariation(true); // gamma+ variation
    HELL::sqmatrix<double>  d1Pv1LL = sxD0LL.DeltaP(as, x);
    //
    sxDv.SetRCvariation(false); // no gamma+ variation (needed as at the moment some global variables are set by this function)
    sxDv.SetRCmode(1); // gamma_qg variation
    double logRv = sxDv.deltaLogR(as, x);
    HELL::sqmatrix<double>   d2Pv2 = sxDv.DeltaP(as, x);
    HELLx::sqmatrix<double> xd2Pv2 = HELLx::sqmatrix<double>(d2Pv2.gg(),d2Pv2.gq(),d2Pv2.qg(),d2Pv2.qq());
    //
    HELLx::sqmatrix<double> d0PLL = sxDLL.DeltaP(as, x, HELLx::LO);
    HELLx::sqmatrix<double> d1PLL = sxDLL.DeltaP(as, x, HELLx::NLO);
    HELLx::sqmatrix<double> d1P   = sxD  .DeltaP(as, x, HELLx::NLO);
    HELLx::sqmatrix<double> d2P   = sxD  .DeltaP(as, x, HELLx::NNLO);
    HELLx::sqmatrix<double> d3P   = sxD  .DeltaP(as, x, HELLx::N3LO);
    HELLx::sqmatrix<double> P3    = sxD.DeltaP(as, x, HELLx::NNLO) - sxD.DeltaP(as, x, HELLx::N3LO);
    sxD.SetRCvar(1); // gamma_+ variation
    HELLx::sqmatrix<double> P3v1  = sxD.DeltaP(as, x, HELLx::NNLO) - sxD.DeltaP(as, x, HELLx::N3LO);
    sxD.SetRCvar(2); // gamma_qg variation
    HELLx::sqmatrix<double> P3v2  = sxD.DeltaP(as, x, HELLx::NNLO) - sxD.DeltaP(as, x, HELLx::N3LO);
    sxD.SetRCvar(0);
    HELLx::sqmatrix<double> d3Pv2 = xd2Pv2 - P3v2;
    HELLx::sqmatrix<double> d3Pv1 = xd2Pv1 - P3v1;
    //
    double xPggLO = x*PggLO(x,as,nf);
    double xPgqLO = x*PgqLO(x,as,nf);
    double xPqgLO = x*PqgLO(x,as,nf);
    double xPqqLO = x*PqqLO(x,as,nf);
    double xPggNLO = htpgg.eval(x); //x*PggNLO(x,as,nf);
    double xPgqNLO = htpgq.eval(x); //x*PgqNLO(x,as,nf);
    double xPqgNLO = x*PqgNLO(x,as,nf);
    double xPqqNLO = htpqq.eval(x); //x*PqqNLO(x,as,nf);
    double xPggNNLO = htpggNNLO.eval(x);
    double xPgqNNLO = htpgqNNLO.eval(x);
    double xPqgNNLO = htpqgNNLO.eval(x);
    double xPqqNNLO = htpqqNNLO.eval(x);
    ofile2 << setw(5)  << as
	   << setw(15) << x
	   << setw(15) << x*d0PLL.gg()
	   << setw(15) << x*d1P.gg()
	   << setw(15) << x*d1P.qg()
	   << setw(15) << xPggLO // 6
	   << setw(15) << xPgqLO
	   << setw(15) << xPqgLO
	   << setw(15) << xPqqLO
	   << setw(15) << xPggNLO // 10
	   << setw(15) << xPgqNLO
	   << setw(15) << xPqgNLO
	   << setw(15) << xPqqNLO
	   << setw(15) << xPggNNLO // 14
	   << setw(15) << xPgqNNLO
	   << setw(15) << xPqgNNLO
	   << setw(15) << xPqqNNLO
	   << setw(15) << CCSS(x, "gg", "c") // 18
	   << setw(15) << CCSS(x, "gg", "u")
	   << setw(15) << CCSS(x, "gg", "d")
	   << setw(15) << CCSS(x, "gq", "c")
	   << setw(15) << CCSS(x, "gq", "u")
	   << setw(15) << CCSS(x, "gq", "d")
	   << setw(15) << CCSS(x, "qg", "c") // 24
	   << setw(15) << CCSS(x, "qg", "u")
	   << setw(15) << CCSS(x, "qg", "d")
	   << setw(15) << CCSS(x, "qq", "c")
	   << setw(15) << CCSS(x, "qq", "u")
	   << setw(15) << CCSS(x, "qq", "d")
	   << setw(15) << x*sqrt(pow(d2P.gg()-d2Pv2.gg(),2)+pow(d2P.gg()-d2Pv1.gg(),2))  // 30
      //   << setw(15) << x*(fabs(d2P.gg()-d2Pv2.gg())+fabs(d2P.gg()-d2Pv1.gg()))  // 30
      //   << setw(15) << x*(d2P.qg()-d2Pv2.qg())  // 31
	   << setw(15) << x*sqrt(pow(d2P.qg()-d2Pv2.qg(),2)+pow(d2P.qg()-d2Pv1.qg(),2))  // 31
	   << setw(15) << x*d1P.gg()-x*d2P.gg()
	   << setw(15) << x*d1P.qg()-x*d2P.qg()
	   << setw(15) << x*sxDLL.DeltaP(as, x, HELLx::LO).gg() - x*sxDLL.DeltaP(as, x, HELLx::NLO).gg()
	   << setw(15) << x*logR0  // 35
	   << setw(15) << x*logRv
	   << setw(15) << x*fabs(d1PLL.gg()-d1Pv1LL.gg())
	   << setw(15) << x*d1PLL.gg()
	   << setw(15) << x*P3.gg()  // 39
	   << setw(15) << x*P3.qg()  // 40
	   << setw(15) << x*sqrt(pow(P3.gg()-P3v2.gg(),2)+pow(P3.gg()-P3v1.gg(),2))  // 41
      //   << setw(15) << x*(P3.qg()-P3v2.qg())
	   << setw(15) << x*sqrt(pow(P3.qg()-P3v2.qg(),2)+pow(P3.qg()-P3v1.qg(),2))  // 41
      // delta3 P
	   << setw(15) << x*d3P.gg()  // 43
	   << setw(15) << x*d3P.qg()
	   << setw(15) << x*sqrt(pow(d3P.gg()-d3Pv2.gg(),2)+pow(d3P.gg()-d3Pv1.gg(),2))
      //   << setw(15) << x*(d3P.qg()-d3Pv2.qg())
	   << setw(15) << x*sqrt(pow(d3P.qg()-d3Pv2.qg(),2)+pow(d3P.qg()-d3Pv1.qg(),2))
	   << endl;
    if(i%20==0) cout << "."; cout.flush();
  }
  cout << endl;


  // Finish time
  gettimeofday(&t1,NULL);
  double t=t1.tv_sec-t0.tv_sec+(t1.tv_usec-t0.tv_usec)*0.000001;
  cout << "Total time: " << t << "s" << endl;

  return 0;

}
