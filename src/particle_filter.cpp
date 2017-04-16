/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  if (!is_initialized) {
    num_particles = 50;

    // setup random vars
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // initialize particles
    for (int i=0; i<num_particles; i++) {
      Particle particle = {
        i,
        dist_x(gen),
        dist_y(gen),
        dist_theta(gen),
        1.0
      };
      particles.push_back(particle);
    }

    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;

  // for every particle
  for (int i=0; i<num_particles; i++) {
    Particle& particle = particles[i];

    // calculate the now position and steering angle
    double x_f, y_f, theta_f;
    if (yaw_rate == 0) {
      x_f = particle.x + velocity * delta_t * cos(particle.theta);
      y_f = particle.y + velocity * delta_t * sin(particle.theta);
      theta_f = particle.theta;
    } else {
      x_f = particle.x + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      y_f = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      theta_f = particle.theta + yaw_rate * delta_t;
    }

    // add noise to the position and steering angle, and update it
    normal_distribution<double> dist_x(x_f, std_pos[0]);
    normal_distribution<double> dist_y(y_f, std_pos[1]);
    normal_distribution<double> dist_theta(theta_f, std_pos[2]);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
  //
  // I chose not to implement this method
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
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
  double total_weight = 0.0;
  // for every particle
  for (int i=0; i<num_particles; i++) {
    Particle& particle = particles[i];

    double weight = 1.0;
    // for each observation
    for (int l=0; l<observations.size(); l++) {
      LandmarkObs obs = observations[l];
      // convert the position to map coordinates
      double map_x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
      double map_y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;

      Map::single_landmark_s closest_landmark;
      double closest_dist;
      // Find the closest landmark to the observation
      for (int map_l=0; map_l<map_landmarks.landmark_list.size(); map_l++) {
        auto landmark = map_landmarks.landmark_list[map_l];
        double landmark_dist = dist(map_x, map_y, landmark.x_f, landmark.y_f);

        if (map_l==0 || landmark_dist < closest_dist) {
          closest_landmark = landmark;
          closest_dist = landmark_dist;
        }
      }
      // Multiply the probability of seeing the closest landmark, with the
      // existing weight, and update the weights
      weight *= (1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1])) *
        exp(-(
          (pow(map_x - closest_landmark.x_f, 2.0) / (2.0 * pow(std_landmark[0], 2.0))) +
          (pow(map_y - closest_landmark.y_f, 2.0) / (2.0 * pow(std_landmark[1], 2.0)))
        ));
    }
    particle.weight = weight;
    total_weight += weight;
  }

  // normalize the weights (this helps with numerical stability)
  for (int i=0; i<num_particles; i++) {
    Particle& particle = particles[i];
    particle.weight = particle.weight / total_weight;
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  std::vector<Particle> resampled_particles;
  discrete_distribution<int> dist_index(0, num_particles);

  // Calculate the max weight, and setup the distributions
  double max_weight = 0;
  for (int i=0; i<num_particles; i++) {
    max_weight = max(particles[i].weight, max_weight);
  }
  uniform_real_distribution<double> dist_beta(0, 2.0 * max_weight);

  // Resmaple from particles using the resampling wheel technique described in
  // lesson 13
  int index = dist_index(gen);
  double beta = 0;
  for (int i=0; i<num_particles; i++) {
    beta += dist_beta(gen);
    while (particles[index].weight < beta) {
      beta = beta - particles[index].weight;
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
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
