#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 50;
    weights.resize(num_particles);
    particles.resize(num_particles);

    // define normal distribution for sensor noise
    random_device rd;
    default_random_engine gen(rd());
    normal_distribution<double> N_x_init(x, std[0]); // simulated estimate from GPS
    normal_distribution<double> N_y_init(y, std[1]);
    normal_distribution<double> N_theta_init(theta, std[2]);

    // init particle
    for(int i =0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = N_x_init(gen);
        particles[i].y = N_y_init(gen);
        particles[i].theta = N_theta_init(gen);
        particles[i].weight = 1.d;
        weights[i] = particles[i].weight;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    for(int i = 0; i < num_particles; i++) {
        double x_new = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
        double y_new = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
        double theta_new = particles[i].theta + yaw_rate * delta_t;

        // define normal distribution for sensor noise
        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> N_x(x_new, std_pos[0]);
        normal_distribution<double> N_y(y_new, std_pos[1]);
        normal_distribution<double> N_theta(theta_new, std_pos[2]);

        particles[i].x = N_x(gen);
        particles[i].y = N_y(gen);
        particles[i].theta = N_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    double sum_w = 0.;

    for(int i = 0; i<num_particles; i++) {
        double w = 1.;

        for(int obs_i = 0; obs_i < observations.size(); obs_i++) {
            LandmarkObs obs = observations[obs_i];

            // landmark transform
            double trans_obs_x = particles[i].x + (obs.x * cos(particles[i].theta)) - (obs.y * sin(particles[i].theta));
            double trans_obs_y = particles[i].y + (obs.x * sin(particles[i].theta)) + (obs.y * cos(particles[i].theta));

            obs.x = trans_obs_x;
            obs.y = trans_obs_y;

            Map::single_landmark_s lm_b;
            double short_lm = numeric_limits<double>::max();
            for(int lm_i = 0; lm_i < map_landmarks.landmark_list.size(); lm_i++) {
                Map::single_landmark_s lm = map_landmarks.landmark_list[lm_i];
                double cur_dist = dist(obs.x, obs.y, lm.x_f, lm.y_f);
                if(cur_dist < short_lm) {
                    short_lm = cur_dist;
                    lm_b = lm;
                }
            }

            double numerator = exp(-0.5 *
                                   (pow((obs.x - lm_b.x_f), 2) / pow(std_landmark[0], 2) +
                                    pow((obs.y - lm_b.y_f), 2) / pow(std_landmark[1], 2)));
            double denominator = 2 * M_PI * std_landmark[0] * std_landmark[1];
            w *= numerator / denominator;
        }
        sum_w += w;
        particles[i].weight = w;
    }

    for (int i = 0; i < num_particles; i++) {
        particles[i].weight /= sum_w;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    random_device rd;
    default_random_engine gen(rd());

    discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> resampled_particles((unsigned long) num_particles);

    for (int i = 0; i < num_particles; i++) {
        resampled_particles[i] = particles[dist_particles(gen)];
    }

    particles = resampled_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
