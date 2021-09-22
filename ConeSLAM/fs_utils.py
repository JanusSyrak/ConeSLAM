import math
import numpy as np
from landmark import *
from copy import deepcopy

# The modules in this code are partly inspired from https://github.com/AtsushiSakai/PythonRobotics/tree/master/SLAM/FastSLAM2

# Four different covariance matrices, one for each cone
R_1 = np.array([0.4812, 0.1162, 0.1162, 0.0440]) * 1e-3
R_1 = R_1.reshape([2,2])

R_2 = np.array([0.0013, 0.0001, 0.0001, 0.0001])
R_2 = R_2.reshape([2,2])

R_3 = np.array([0.4242, 0.0365, 0.0365, 0.0408]) * 1e-3
R_3 = R_3.reshape([2,2])

R_4 = np.array([0.6358, 0.0031, 0.0031, 0.0871]) * 1e-3
R_4 = R_4.reshape([2,2])

# Pooled covariance matrix
R = np.array([0.9309, 0.1313, 0.1313, 0.0918]) * 1e-3
R = R.reshape([2, 2])

P_cov = np.array([5.77891, 0, 0, 0.599415307]) * 1e-3
P_cov = P_cov.reshape([2, 2])

def getCovMatrix(class_index):
    covs = [R_1, R_2, R_3, R_4]

    return covs[class_index]


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def homog_to_rb(homog_transform):
    r = math.sqrt(homog_transform[0, 2] ** 2 + homog_transform[1, 2] ** 2)
    bearing = math.asin(homog_transform[0,1])
    return np.array([r, bearing])


def frame_to_rb(frame):
    obs = []
    for pt in frame:
        r = math.sqrt(pt[0] ** 2 + pt[1] ** 2) 
        bearing = math.atan2(pt[1], pt[0]) - math.pi/2
        obs.append([r, bearing, pt[2]])

    return np.array(obs)


def rb_to_xy(pt):
    fixed_angle = pi_2_pi(pt[1] + math.pi/2)
    x = pt[0] * math.cos(fixed_angle)
    y = pt[0] * math.sin(fixed_angle)
    return np.array([x, y])


def compute_jacobians(particle, lm):
    dx = lm.x - particle.x
    dy = lm.y - particle.y
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)

    # zHat denotes our estimate of the location of the landmarks of the particle
    zHat = np.array(
        [d,
         pi_2_pi(math.atan2(dy, dx) - particle.yaw)])

    # G is the derivative of the measurement model wrt the landmark of the particle
    G = np.array([[dx / d,   dy / d],
                  [-dy / (dx ** 2 + dy ** 2),
                   dx/(dx ** 2 + dy ** 2)]])

    # Gs is the derivative of the measurement model wtr to the particle state
    # This is only used in FastSLAM 2.0 proposal sampling
    Gs = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d2, -dx / d2, -1.0]])

    Zn = G @ lm.cov @ G.T + R

    return zHat, G, Zn, Gs


def add_new_landmark(z, particle, color_index):
    new_lm = Landmark()
    new_lm.colorcount[int(color_index - 1)] += 1
    Ht = particle.getAbsolutePosition(z)
    new_lm.updatePosition(Ht)
    _, G, _, _ = compute_jacobians(particle, new_lm)

    # Covariance of the landmark
    sigma = np.linalg.inv(G.T @ np.linalg.inv(R) @ G)

    new_lm.cov = sigma
    
    new_lm.addClassification(color_index, z[0])
    particle.lm = np.append(particle.lm, new_lm)


def compute_weight(particle, lm, z):
    zHat, G, Zn, _ = compute_jacobians(particle, lm)

    dz = z - zHat
    dz[1] = pi_2_pi(dz[1])

    try:
        invZn = np.linalg.inv(Zn)
    except np.linalg.linalg.LinAlgError:
        return 1.0

    num = math.exp(-0.5 * dz.T @ invZn @ dz)
    den = math.sqrt(2.0 * math.pi * np.linalg.det(Zn))

    w = num / den

    return w

def update_with_observation(z, particles):
    for iz in range(z.shape[0]):
        for ip in range(len(particles)):
            # Need to use the absolute positions to compare the Euclidean distances
            absolute_measurement = particles[ip].getAbsolutePosition(z[iz, :2])
            current_particle = particles[ip]

            # We use the Euclidean distance from the landmark to the observation in order to determine whether to add or update landmark
            # An array with distances from the current observations to all the landmarks of the particle
            dists = []
            for lm in current_particle.lm:
                dist = math.sqrt((absolute_measurement[0] - lm.x) ** 2 + (absolute_measurement[1] - lm.y) ** 2)
                dists.append(dist)
            smallest_dist = 100000
            if current_particle.lm.shape[0] > 0:
                best_dist_index = np.argmin(dists)
                smallest_dist = dists[best_dist_index]
                
            distance_threshold = 1.0

            # Unknown landmark
            color_index = z[iz, 2]
            R = getCovMatrix(color_index)
            if smallest_dist > distance_threshold:
                add_new_landmark(z[iz, :2], current_particle, color_index)

            # Known landmark
            else:
                lm = current_particle.lm[best_dist_index]
                w = compute_weight(particles[ip], lm, z[iz, :2])
                particles[ip].w *= w
                update_landmark(particles[ip], lm, z[iz, :2], color_index)
                proposal_sampling(particles[ip], lm, z[iz, :2])
    return

# The FastSLAM 2.0 proposal sampling. This takes the observations into account
def proposal_sampling(particle, lm, z):
    # State
    x = np.array([particle.x, particle.y, particle.yaw]).reshape(3, 1)
    zHat, G, Zn, Gs = compute_jacobians(particle, lm)

    invZn = np.linalg.inv(Zn)
    dz = z - zHat
    dz[1] = pi_2_pi(dz[1])
    dz = dz.reshape((2,1))
    invP = np.linalg.inv(particle.P)

    particle.P = np.linalg.inv(Gs.T @ invZn @ Gs + invP)  # proposal covariance
    x += particle.P @ Gs.T @ invZn @ dz  # proposal mean
    particle.x = x[0, 0]
    particle.y = x[1, 0]
    particle.yaw = x[2, 0]


def update_landmark(particle, lm, z, color_index):
    lm_mu = lm.XY()
    lm_cov = lm.cov
    lm.colorcount[int(color_index - 1)] += 1

    lm.addClassification(color_index, z[0])


    zHat, G, Zn, Gs = compute_jacobians(particle, lm)

    dz = z - zHat
    dz[1] = pi_2_pi(dz[1])

    Kt = lm_cov @ G.T @ np.linalg.inv(Zn)
    new_mu = lm_mu + Kt @ dz
    lm.updatePosition(new_mu)

    new_cov = (np.identity(2) - Kt @ G) @ lm.cov
    lm.cov = new_cov

def sample_particles(particles, input_estimate):
    x = input_estimate[0, 2]
    y = input_estimate[1, 2]
    input_estimate[0,2] = y
    input_estimate[1,2] = x
    for i in range(len(particles)):
        particle_homo = particles[i].getHomogeneousTransform()
        new_transform = particle_homo @ input_estimate
        particles[i].x = new_transform[0,2]
        particles[i].y = new_transform[1,2]

        
        sign = math.atan2(-math.asin(new_transform[0,1]), math.acos(new_transform[0,0]))
        angle = math.acos(new_transform[0, 0])
        if sign < 0:
            angle *= -1
        particles[i].yaw = angle

        particles[i].transformations.append(particles[i].getHomogeneousTransform())

        mean = [0,0]

        # Compute three different noise values, x, y and yaw, for the sampling
        range_noise, angle_noise = np.random.multivariate_normal(mean, P_cov).T
        y_noise, _ = np.random.multivariate_normal(mean, P_cov).T

        noise_x = range_noise
        particles[i].x += noise_x
        particles[i].y += y_noise
        particles[i].yaw += angle_noise


def normalize_weight(particles, NO_PARTICLES):
    sum_w = sum([p.w for p in particles])

    try:
        for i in range(NO_PARTICLES):
            particles[i].w /= sum_w
    except ZeroDivisionError:
        for i in range(NO_PARTICLES):
            particles[i].w = 1.0 / NO_PARTICLES

        return particles

    return particles


# low variance re-sampling
def resampling(particles):
    NO_PARTICLES = len(particles)
    NTH = NO_PARTICLES / 1.5
    particles = normalize_weight(particles, NO_PARTICLES)
    particle_weights = np.array([p.w for p in particles])

    effective_particle_number = 1.0 / (particle_weights @ particle_weights.T)

    if effective_particle_number < NTH:
        print("RESAMPLING")
        particle_indices = []
        w_cum = np.cumsum(particle_weights)
        r = np.random.random() / NO_PARTICLES
        index = 0
        for j in range(NO_PARTICLES):
            U = r + j / NO_PARTICLES
            while U > w_cum[index]:
                index += 1
            particle_indices.append(index)

        tmp_particles = deepcopy(particles[:])
        for i in range(len(particle_indices)):
            particles[i].x = tmp_particles[particle_indices[i]].x
            particles[i].y = tmp_particles[particle_indices[i]].y
            particles[i].yaw = tmp_particles[particle_indices[i]].yaw

            tmp_lm = deepcopy(tmp_particles[particle_indices[i]].lm[:])
            particles[i].lm = tmp_lm
            particles[i].w = 1.0 / NO_PARTICLES

    lengths = [len(particle.lm) for particle in particles]
    return particles
