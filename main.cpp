#include <triqs/mc_tools/random_generator.hpp>
#include <triqs/mc_tools/mc_generic.hpp>
#include <triqs/utility/callbacks.hpp>
#include <triqs/arrays.hpp>
#include <triqs/statistics.hpp>
#include <fstream>

#define PI 3.14159265

using namespace triqs::arrays;
using namespace triqs::statistics;

/**************
 * config
 **************/



struct configuration {
    // N is the linear size of spin matrix, M the total magnetization,
    // beta the inverse temperature, J the coupling,
    // field the magnetic field and energy the energy of the configuration
    int N,iter,iter_step;
    double beta,beta_step, J, K, B, D;

    // the chain of spins: true means "up", false means "down"
    array<double, 3> lattice;

    // constructor
    configuration(int N_,int iter_step_, double beta_,double beta_step_, double J_, double K_, double B_, double D_) :
            N(N_),iter(0),iter_step(iter_step_), beta(beta_),beta_step(beta_step_), J(J_), K(K_), B(B_), D(D_), lattice(N, N, 3) { lattice() = 0; }
};

/**************
 * move
 **************/

// A move flipping a random spin
struct flip {
    configuration *config;
    triqs::mc_tools::random_generator &RNG;

    struct site { int i, j; };//small struct storing indices of a given site
    site s;
    double delta_energy = 0;
    array<double, 1> spin;
    int index;


    // constructor
    flip(configuration &config_, triqs::mc_tools::random_generator &RNG_) :
            config(&config_), RNG(RNG_),spin(3) {}

    // find the neighbours with periodicity
    std::vector<site> neighbors(site s, int N){
        std::vector<site> nns(4);
        int counter=0;
        for(int i=-1;i<=1;i++){
            for(int j=-1;j<=1;j++){
                if ((i==0) != (j==0)) //xor
                    nns[counter++] = site{(s.i+i+N)%N, (s.j+j+N)%N};
            }
        }
        return nns;
    }

    double attempt() {
        // pick a random site
        index = RNG(config->N * config->N);
        s = {index % config->N, index / config->N};


        double theta = RNG(180);
        double phi = RNG(360);
        delta_energy = 0;
        spin(0) = sin(theta / 180.* PI) * cos(phi / 180.* PI);
        spin(1) = sin(theta / 180.* PI) * sin(phi / 180.* PI);
        spin(2) = cos(theta / 180.* PI);
        auto nns = neighbors(s, config->N);

        for (auto &nn:nns) {
            delta_energy += config->J * sum(config->lattice(nn.i, nn.j, range()) * (config->lattice(s.i, s.j, range()) - spin));
            if((nn.i==s.i)&&(nn.j<s.j))delta_energy+=config->D*((config->lattice(s.i, s.j, 1)-spin(1))*config->lattice(nn.i,nn.j,2)-(config->lattice(s.i, s.j, 2)-spin(2))*config->lattice(nn.i,nn.j,1));
            if((nn.j==s.j)&&(nn.i<s.i))delta_energy+=config->D*((config->lattice(s.i, s.j, 0)-spin(0))*config->lattice(nn.i,nn.j,2)-(config->lattice(s.i, s.j, 2)-spin(2))*config->lattice(nn.i,nn.j,0));
        }
        delta_energy += config->K * (config->lattice(s.i, s.j, 2) * config->lattice(s.i, s.j, 2) - spin(2) * spin(2));
        delta_energy += config->B * (config->lattice(s.i, s.j, 2) - spin(2));

        // return Metroplis ratio
        return std::exp(-config->beta * delta_energy);
    }

    // if move accepted just flip site and update energy and magnetization
    double accept() {
        config->lattice(s.i, s.j, range()) = spin;
        config->iter+=1;
        if(config->iter>config->iter_step){
            config->iter=0;
            config->beta*=config->beta_step;
        }
        return 1.0;
    }

    // nothing to do if the move is rejected
    void reject() {}
};


/**************
 * measure
 **************/
struct compute_m {

    configuration *config;
    double Z;
    compute_m(configuration &config_) : config(&config_), Z(1){}
    // accumulate Z and magnetization
    void accumulate(double sign) {  }
    // get final answer M / (Z*N)
    void collect_results(triqs::mpi::communicator c) { }
};

int main(int argc, char **argv) {

    // initialize mpi
    triqs::mpi::environment env(argc, argv);
    triqs::mpi::communicator world;
    // parameters of the model
    int size = 8;
    int iter_step=10000;
    int n_warmup_cycles = 1000000;
    double beta = 1;
    double beta_step=1.1;
    double J = -1;
    double K = 0.1;
    double B = 0;
    double D = 0;
    if(argc==10){
        n_warmup_cycles = atoi(argv[6]);
        beta =  atof(argv[7]) ;
        beta_step =  atof(argv[8]) ;
        iter_step = atoi(argv[1]);
        size = atoi(argv[9]);
        J =  atof(argv[5]) ;
        K =  atof(argv[4]) ;
        B =  atof(argv[2]) ;
        D =  atof(argv[3]) ;
    }
    if (world.rank() == 0) std::cout << std::endl;
    if (world.rank() == 0) std::cout << "\tAnnealing parameters n_cycles = " << n_warmup_cycles << ", beta = " << beta << ", beta_step = " << beta_step << ", iter_step = " <<iter_step<<  std::endl;
    if (world.rank() == 0) std::cout << "\t2D Heisenberg with size = "<<size<<", J = "<< J << ", K = " << K << ", B = " << B << ", D = " << D  << std::endl;

    // Prepare the MC parameters
    int n_cycles = 0;
    int length_cycle = 1;

    std::string random_name = "";
    int random_seed = 374982 + world.rank() * 273894;
    int verbosity = (world.rank() == 0 ? 2 : 0);

    // Construct a Monte Carlo loop
    triqs::mc_tools::mc_generic<double> Skyrmion(n_cycles, length_cycle, n_warmup_cycles,random_name, random_seed, verbosity);


    // construct configuration
    configuration config(size,iter_step, beta,beta_step, J, K, B, D);

    // add moves and measures
    Skyrmion.add_move(flip(config, Skyrmion.get_rng()), "spin flip");
    Skyrmion.add_measure(compute_m(config), "measure magnetization");

    // Run and collect results
    Skyrmion.warmup_and_accumulate(n_warmup_cycles, n_cycles, length_cycle, triqs::utility::clock_callback(-1));
    if (world.rank() == 0) {
        std::cout <<  "\tBeta Final:\t"<< config.beta << std::endl;
        std::ofstream outfile("magnetization.dat");
        outfile << config.lattice(range(),range(),0) <<std::endl;
        outfile << config.lattice(range(),range(),1) <<std::endl;
        outfile << config.lattice(range(),range(),2) <<std::endl;
        outfile.close();
    }
    if (world.rank() == 0) std::cout <<"\tDone\t"<< std::endl;
    return 0;
}
