import torch
import numpy as np
import math
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import math
from sklearn.cluster import KMeans
from torch.distributions import kl_divergence
import argparse
import matplotlib.pyplot as plt
import datetime
import time

torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device: ', device)

# parameter in terminal
params = argparse.ArgumentParser()
params.add_argument('-num_iterations', type=int, default=10000, help='iteration time ')
params.add_argument('-learning_rate', type=float, default=0.001, help='learning rate')
params.add_argument('-output_scale_generate', type=float, default=0.8, help='output scale for generating Y')
params.add_argument('-length_scale_generate_1', type=float, default=1.0, help='length scale 1 inverse for generating Y')
params.add_argument('-length_scale_generate_2', type=float, default=1.5, help='length scale 2 inverse for generating Y')
params.add_argument('-noise_generate', type=float, default=30., help='noise std inverse for generating Y')
params.add_argument('-output_scale_model', type=float, default=0.5, help='initialized output scale for model')
params.add_argument('-length_scale_model_1', type=float, default=0.5, help='initialized length scale 1 inverse for model')
params.add_argument('-length_scale_model_2', type=float, default=0.5, help='initialized length scale 2 inverse for model')
params.add_argument('-noise_model', type=float, default=8., help='initialized noise std inverse for model')
params.add_argument('-data_points', type=int, default=300, help='number of data points')
args = params.parse_args()

num_iterations = args.num_iterations
learning_rate = args.learning_rate
output_scale_sqr_generate = args.output_scale_generate
length_scale_inv_generate_1 = args.length_scale_generate_1
length_scale_inv_generate_2 = args.length_scale_generate_2
noise_inv_generate = args.noise_generate
output_scale_model = args.output_scale_model
length_scale_inv_model_1 = args.length_scale_model_1
length_scale_inv_model_2 = args.length_scale_model_2
noise_inv_model = args.noise_model
data_points = args.data_points

# ARD kernel
def squared_exponential_kernel(X1, X2, length_scale, output_scale):
    # print('X1',X1)
    # print('ls',length_scale)
    # print('outputscale',output_scale)
    pairwise_distances = torch.cdist(X1 * torch.sqrt(length_scale), X2 * torch.sqrt(length_scale), p=2).pow(2)
    # print('pairwise_distances',pairwise_distances)
    kernel_values = output_scale * torch.exp(-0.5 * pairwise_distances)
    return kernel_values

# RBF kernel
# def squared_exponential_kernel(x1, x2, length_scale, output_scale):
#     pairwise_sq_dists = torch.sum((x1[:, None] - x2) ** 2, dim=-1)
#
#     return output_scale ** 2 * torch.exp(-pairwise_sq_dists / (2 * (length_scale ** 2)))


# def initialize_inducing_inputs(X, M, random_seed=42):
#     # print('X', X)
#     torch.manual_seed(random_seed)
#     indices = torch.randperm(X.size(0))[:M]
#     inducing_inputs = X[indices]
#     # print('inducing points', inducing_inputs)
#     return inducing_inputs

def initialize_inducing_inputs(X, M, seed=42):
    # print('X', X)
    torch.manual_seed(seed)

    # inducing point = kmeans method
    kmeans = KMeans(n_clusters=M, random_state=seed)
    # print('kmeans',kmeans)
    kmeans.fit(X)
    # print('kmeans.fit(X)', kmeans.fit(X))
    inducing_inputs = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    return inducing_inputs

def initialize_pca_package(Y, latent_dim, seed):
    torch.manual_seed(seed)
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

def initialize_pca_customize(Y, N, D, Q):
    # standalization
    mean = torch.mean(Y, dim=0)
    # std = torch.std(Y, dim=0)
    # std_matrix = (Y - mean) / std
    std_matrix = Y - mean
    # covariance
    cov_matrix = (std_matrix.T @ std_matrix) / (Y.size(0) - 1)
    # choose eigenvalue and eigenvector
    eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
    sorted_indices_descend = torch.argsort(eigenvalues.real, descending=True)
    eigenvectors_indices = eigenvectors[:, sorted_indices_descend].real
    eigenvalues_indices = eigenvalues[sorted_indices_descend].real
    # generate principle component
    principal_components = eigenvectors_indices[:, :Q]
    # final matrix
    final_matrix = (principal_components.T @ std_matrix.T).T
    return final_matrix

def prior_x_mean_var(prior_x_mean, prior_x_var, N_Nstar, N, Q):
    mean_train_test_X = (torch.ones(N_Nstar, Q) * prior_x_mean).to(device)
    var_train_test_X = (torch.ones_like(mean_train_test_X)*prior_x_var).to(device)
    std_train_test_X = torch.sqrt(var_train_test_X)
    mean_p_x = mean_train_test_X[:N, :].to(device)  # size(N,2)
    var_p_x = var_train_test_X[:N, :].to(device)
    std_p_x = std_train_test_X[:N, :].to(device)
    return mean_train_test_X,std_train_test_X,mean_p_x, std_p_x, var_p_x

def train_Y(X, D, output_scale_layer1, length_scale_layer1, noise_variance, seed,trainY_mode):
    '''

    Args:
        X:
        D:
        output_scale_layer1:
        length_scale_layer1:
        noise_variance:
        train_mode: "duplicate", "different"

    Returns:

    '''
    print('trainY mode',trainY_mode)
    torch.manual_seed(seed)
    # training input function value
    # P(F(X)) = N( m(X), K1(X,X)=K_nn ) mean and variance
    # m(X) = 0
    mean_vector_X = torch.zeros(len(X), 1).to(device)
    K_nn = squared_exponential_kernel(
        X, X, length_scale_layer1, output_scale_layer1
    )

    p_Y = torch.distributions.MultivariateNormal(mean_vector_X.reshape(1, -1).squeeze(0),
                                                 K_nn + (1 / noise_variance) * torch.eye(len(X)).to(device))
    # duplicate Y for D times
    if trainY_mode == "duplicate":
        p_Y_sample_2 = p_Y.sample((1,)).T
        Y = p_Y_sample_2.repeat(1, D)
    if trainY_mode == "different":
        p_Y_sample_2 = p_Y.sample((D,))
        Y = p_Y_sample_2.T
    print('Y', Y)

    return Y

def inputX_observationY(output_scale_sqr_generate,
                        length_scale_inv_generate_1,
                        length_scale_inv_generate_2,
                        noise_inv_generate,
                        mean_train_test_X,
                        std_train_test_X,
                        D, N, Q, seed):
    torch.manual_seed(seed)
    output_scale_layer1_generate = torch.tensor([output_scale_sqr_generate]).to(device)
    length_scale_layer1_inv_generate = torch.tensor([length_scale_inv_generate_1, length_scale_inv_generate_2]).to(device)
    noise_std_inv_generate = torch.tensor([noise_inv_generate]).to(device)
    train_test_X_prior = torch.distributions.Normal(mean_train_test_X, std_train_test_X)
    samples_train_test_X = train_test_X_prior.sample().to(device)
    # samples_train_test_X = torch.randn(N, Q)
    print('GT length scale',length_scale_layer1_inv_generate)
    print('GT output scale', output_scale_layer1_generate)
    print('GT beta', noise_std_inv_generate)
    print('samples_train_test_X', samples_train_test_X)

    # "duplicate", "different"
    trainY_mode = "different"
    Y_train_test = train_Y(samples_train_test_X, D, output_scale_layer1_generate, length_scale_layer1_inv_generate,
                           noise_std_inv_generate, seed, trainY_mode=trainY_mode)
    Y_train_test = Y_train_test.double()
    # print('Y_train_test',Y_train_test)
    # print('Y_train_test',Y_train_test.shape)
    Y = Y_train_test[:N, :].to(device)
    # Y = torch.tensor([[-1.0226],
    #     [-0.2527],
    #     [-0.7900],
    #     [ 1.0774],
    #     [ 1.5436]]).double()
    return Y, samples_train_test_X, trainY_mode

def inducing_generate_Z(samples_train_test_X, M, seed,mode="sequential"):
    '''

    Args:
        samples_train_test_X:
        M:
        mode: "sequential", "random", "kmeans", "pca"
    Returns:

    '''
    torch.manual_seed(seed)
    print('inducing mode',mode)
    if mode == "sequential":
        Z = samples_train_test_X[0:M, :]
    if mode == "kmeans":
        Z = initialize_inducing_inputs(samples_train_test_X, M, seed)

    # Z and P(U) = N( m(Z), K1(Z,Z) = K_mm )
    # Z = torch.tensor([[-0.7056,  0.6741],
    #         [-0.5454,  0.9107],
    #         [ 1.0682,  0.1424]])
    Z = Z.double().to(device)
    print('Z', Z)
    return Z

def q_X_var_mean(samples_train_test_X, Y, Q, N, seed, mode):
    '''

    Args:
        samples_train_test_X:
        Y:
        Q:
        N:
        mode: "fixed", "pca-package", "pca-customize", "input", "random"

    Returns:

    '''
    torch.manual_seed(seed)
    # q(Xq) = N(mean_q(Xq), var_q(Xq))
    print('q(X) mode:',mode)
    if mode == "fixed":
        mean_q_x = torch.nn.Parameter((torch.ones(N, Q) * 0.1).double().to(device), requires_grad=True)
    if mode == "pca-package":
        mean_q_x = torch.nn.Parameter(initialize_pca_package(Y, Q, seed).double().to(device), requires_grad=True)
    if mode == "pca-customize":
        mean_q_x = torch.nn.Parameter(initialize_pca_customize(Y, N, D, Q).double().to(device), requires_grad=True)
    if mode == "input":
        mean_q_x = torch.nn.Parameter(samples_train_test_X.double().to(device), requires_grad=True)
    if mode == "random":
        mean_q_x = torch.nn.Parameter(torch.randn(N, Q).double().to(device), requires_grad=True)

    log_sigma_q_x = torch.nn.Parameter((torch.ones(N, Q) * 1.0).double().to(device), requires_grad=True)
    mean_q_x_save = mean_q_x.detach().clone().to(device)
    return mean_q_x, mean_q_x_save, log_sigma_q_x

def KL_divergence(mean_p_x, mean_q_x, var_p_x, var_q_x):
    '''
    KL divergence KL(q(X)||p(x))
    KL(q(x) || p(x)) = 0.5 * [ ln(det(Σ_p) / det(Σ_q)) + tr(Σ_p⁻¹ * Σ_q) + (μ_p - μ_q)ᵀ * Σ_p⁻¹ * (μ_p - μ_q) - k]

    '''

    q_x = torch.distributions.Normal(mean_q_x, torch.sqrt(var_q_x))
    p_x = torch.distributions.Normal(mean_p_x, torch.sqrt(var_p_x))
    kl_per_latent_dim = kl_divergence(q_x, p_x).sum(axis=0)
    KL_q_p = kl_per_latent_dim.sum()

    return KL_q_p

def fi0_function(N, output_scale_layer1):
    '''
    fi(0) is a scaler,
    fi(0) = sum_n^N(fi_n(0)) = N * outputscale**2
    '''
    return N * (output_scale_layer1)

def fi1_function(N, M, Q, output_scale_layer1, w_q, mean_q_x, var_q_x, Z):
    '''
    fi_1 is a N*M matrix
    fi_1_nm =
    '''
    # print('Z',Z)
    # print('mean_q_x',mean_q_x)
    fi_1 = torch.zeros(N, M).double().to(device)
    fraction_multiply_term = 1
    for q in range(Q):
        miu_nq_minus_zmq = torch.cdist(mean_q_x[:,q].reshape(-1, 1), Z[:, q].reshape(-1, 1), p=2).pow(2)
        # print('miu_nq_minus_zmq',miu_nq_minus_zmq)
        # print('miu_nq_minus_zmq', miu_nq_minus_zmq.shape)
        exp_term = - 0.5 * (w_q[q] * miu_nq_minus_zmq) / (w_q[q] * var_q_x[:, q].reshape(-1, 1) + 1)
        # print('exp_term',exp_term)
        denominator = torch.sqrt(w_q[q] * var_q_x[:, q].reshape(-1, 1) + 1)
        fraction_term_q = torch.exp(exp_term) / denominator
        # print('fraction_term_q', fraction_term_q)
        fraction_multiply_term = fraction_multiply_term * fraction_term_q
    fi_1 = (output_scale_layer1) * fraction_multiply_term
    # print('fi_1',fi_1)


    return fi_1

def fi2_function(N, M, Q, output_scale_layer1, w_q, mean_q_x, var_q_x, Z):
    '''
    fi2 is a M*M matrix
    fi2 = sum^N_n(fi_2_n)
    (fi_2_n)mm' =
    '''
    i = 0
    fi_2 = torch.zeros(M, M).double().to(device)
    for n in range(N):
        # fi_2_n = torch.zeros(M, M).double().to(device)
        fraction_multiply_term = 1
        for q in range(Q):
            z_mq_minus_zmprimeq = torch.cdist(Z[:, q].reshape(-1,1) , Z[:, q].reshape(-1,1) , p=2).pow(2)
            # print('z_mq_minus_zmprimeq',z_mq_minus_zmprimeq)
            # print('z_mq_minus_zmprimeq', z_mq_minus_zmprimeq.shape)
            exp_term1 = - 0.25 * w_q[q] * z_mq_minus_zmprimeq
            Zq = Z[:, q].reshape(-1,1)
            #z_q = 0.5 * (Z[m, q] + Z[m_prime, q])
            z_mq_add_zmprimeq = torch.sum((Zq[:, None] + Zq), dim=-1) *0.5
            # print('z_mq_add_zmprimeq ',z_mq_add_zmprimeq )
            # print('z_mq_add_zmprimeq ', z_mq_add_zmprimeq.shape)

            exp_term2 = - (w_q[q] * ((mean_q_x[n, q] - z_mq_add_zmprimeq) ** 2)) / (
                    2 * w_q[q] * var_q_x[n, q] + 1)
            denominator = torch.sqrt(2 * w_q[q] * var_q_x[n, q] + 1)
            fraction_term_q = torch.exp(exp_term1 + exp_term2) / denominator
            fraction_multiply_term = fraction_multiply_term * fraction_term_q
        fi_2_n= (output_scale_layer1 ** 2) * fraction_multiply_term
        # fi_2_n[m, m_prime] = fi_2_n_mm
        # print('fi_2_n',fi_2_n)

        fi_2 = fi_2 + fi_2_n

    return fi_2

def W_term(N, beta, fi_1, fi_2, K_mm):
    '''
    W = beta *I_N - beta^2 * fi_1 * (beta * fi_2 + K_mm)^(-1) * (fi_1)^T
    '''
    # print('beta',beta)
    # print('fi_1',fi_1)
    W = beta * torch.eye(N).double().to(device) - (beta ** 2) * fi_1 @ torch.inverse(beta * fi_2 + K_mm) @ (fi_1.T)
    # print('torch.inverse(beta * fi_2 + K_mm)',torch.inverse(beta * fi_2 + K_mm))
    return W

def log_term_faction_function(N, beta, fi_2, K_mm):
    '''
    log term in loss function
    '''
    log_term1 = 0.5 * N * torch.log(beta) + 0.5 * torch.logdet(K_mm)
    log_term2 = -0.5 * N * torch.log(torch.tensor(2 * math.pi).to(device)) - 0.5 * torch.logdet((beta * fi_2 + K_mm))

    log_term = log_term1 + log_term2

    # print('log_term',log_term)
    # print(log_term)
    return log_term

def log_exp_term_function(yd, W):
    log_exp_term = - 0.5 * yd.T @ W @ yd
    return log_exp_term

def trace_part_function(beta, fi_0, fi_2, K_mm):
    '''
       trace part in loss function
       trace part  = -(beta * fi_0) / 2 + (bets / 2) * trace(K_mm.inverse @ fi_2)
       '''
    trace_part_1 = -0.5 * beta * fi_0
    # print('K_mm',K_mm)
    trace_part_2 = 0.5 * beta * torch.trace(torch.inverse(K_mm) @ fi_2)
    return trace_part_2 + trace_part_1

def loss_variational_bound(N, M, Q, D, Y, Z,
                           length_scale_layer1, output_scale_layer1, noise_variance,
                           mean_p_x, mean_q_x,
                           var_p_x, log_sigma_q_x,
                           jitter_param
                           ):

    # q(Xq) = N(mean_q(Xq), var_q_x)
    var_q_x = log_sigma_q_x
    # print('var_q_x',var_q_x)

    mean_vector_Z = torch.zeros(len(Z), 1).double().to(device)
    K_mm = squared_exponential_kernel(
        Z, Z, length_scale_layer1, output_scale_layer1
    ).double()
    # print('K_mm',K_mm)
    K_mm += jitter_param * torch.eye(len(Z)).to(device)
    # print('K_mm_inv', K_mm)
    # F(q) = sum_D( Fd(q) ) - KL(q(X)||p(X))
    # kl(q(X)||p(X))
    kl_q_p = KL_divergence(mean_p_x, mean_q_x, var_p_x, var_q_x)
    Fd_q_sum = 0.
    # Fd(q)
    # fi(0), fi(1), fi(2)
    w_q = length_scale_layer1
    beta = noise_variance
    fi_0 = fi0_function(N, output_scale_layer1)
    fi_1 = fi1_function(N, M, Q, output_scale_layer1, w_q, mean_q_x, var_q_x, Z)  # fi_1 is a N*M matrix
    fi_2 = fi2_function(N, M, Q, output_scale_layer1, w_q, mean_q_x, var_q_x, Z)  # fi2 is a M*M matrix
    # print('fi_0', fi_0)
    # print('fi_1', fi_1)
    # print('fi_2', fi_2)
    for d in range(D):

        W = W_term(N, beta, fi_1, fi_2, K_mm)
        # print('W',W)
        # print('yd', Y[:, d].reshape(-1, 1))
        yd = Y[:, d].reshape(-1, 1)

        log_exp_term = log_exp_term_function(yd, W)
        Fd_q = log_exp_term
        Fd_q_sum = Fd_q_sum + Fd_q
    trace_part = trace_part_function(beta, fi_0, fi_2, K_mm)
    log_fraction_term = log_term_faction_function(N, beta, fi_2, K_mm)
    Fd_q_sum = Fd_q_sum + (trace_part + log_fraction_term) * D
    print('Fd_q_sum',Fd_q_sum)
    print('klq_p',kl_q_p)
    F_q = Fd_q_sum - kl_q_p
    # F_q = Fd_q_sum
    return F_q / N

def prediction_function(N, M, Q, X_test,Y, Z,
                        length_scale_layer1_optimize, output_scale_layer1_optimize,
                        w_q, beta, d,
                        mean_q_x_optimize, var_q_x_optimize):
    '''
    m(fd*) = K_n*m (fi_2 + beta^(-1) * Kmm)^(-1) * fi_1^(T) * yd
           = K_n*m * beta *(beta * fi_2 +  Kmm)^(-1) * fi_1^(T) * yd

    cov(fd*) = K_nn_star - K_n*m @ (Kmm^(-1) - (beta * fi_2 + Kmm)^(-1)) @ Kmn*
    '''

    yd = Y[:, d].reshape(-1, 1)
    K_nstar_m = squared_exponential_kernel(
        X_test, Z, length_scale_layer1_optimize, output_scale_layer1_optimize
    ).double()
    K_mm_optimize = squared_exponential_kernel(
        Z, Z, length_scale_layer1_optimize, output_scale_layer1_optimize
    ).double()
    K_nn_star = squared_exponential_kernel(
        X_test, X_test, length_scale_layer1_optimize, output_scale_layer1_optimize
    ).double()
    fi_1 = fi1_function(N, M, Q, output_scale_layer1_optimize, w_q, mean_q_x_optimize, var_q_x_optimize, Z)
    fi_2 = fi2_function(N, M, Q, output_scale_layer1_optimize, w_q, mean_q_x_optimize, var_q_x_optimize, Z)

    # m(fd*)
    # mean_fd_star = K_nstar_m @ torch.inverse(fi_2 + (beta ** (-1)) * K_mm_optimize) @ fi_1.T @ yd
    mean_fd_star = beta * K_nstar_m @ torch.inverse(beta * fi_2 + K_mm_optimize) @ fi_1.T @ yd
    # print('mean_fd_star', mean_fd_star)

    # cov(fd*)
    covar_fd_star = K_nn_star - \
                    K_nstar_m @ (torch.inverse(K_mm_optimize) - torch.inverse(beta * fi_2 + K_mm_optimize)) @ (
                        K_nstar_m.T)
    # print('covar_fd_star', covar_fd_star)
    return mean_fd_star, covar_fd_star

def prediction_train_X(samples_train_test_X,
                       length_scale_inv_layer1,
                       output_scale_layer1,
                       noise_std_inv,
                       Y):
    # prediction
    X_test = samples_train_test_X

    length_scale_layer1_optimize = length_scale_inv_layer1.detach()
    output_scale_layer1_optimize = output_scale_layer1.detach()
    noise_variance_optimize = noise_std_inv.detach()
    # noise_term = noise_variance_optimize ** (-2)

    K_train_train = squared_exponential_kernel(samples_train_test_X, samples_train_test_X,
                                               length_scale_layer1_optimize,
                                               output_scale_layer1_optimize)
    K_train_test = squared_exponential_kernel(samples_train_test_X, X_test, length_scale_layer1_optimize,
                                              output_scale_layer1_optimize)
    K_test_test = squared_exponential_kernel(X_test, X_test, length_scale_layer1_optimize,
                                             output_scale_layer1_optimize)

    K_train_train += (torch.eye(X_test.size(0)).to(device)) * noise_variance_optimize ** 2

    K_inv = torch.inverse(K_train_train)
    mean = K_train_test.t() @ K_inv @ Y
    print('print the predicted mean', mean)
    return mean


def main_train(N_Nstar, N, N_star, Q, D, M,
               prior_x_mean, prior_x_var, jitter_value,
               output_scale_sqr_generate,
               length_scale_inv_generate_1,
               length_scale_inv_generate_2,
               noise_inv_generate,
               output_scale_model,
               length_scale_inv_model_1,
               length_scale_inv_model_2,
               noise_inv_model,
               seed,
               learning_rate,
               num_iterations):

    start_time = time.time()

    torch.manual_seed(seed)
    # mean and variance of prior X
    print('learning rate',learning_rate)
    print('iteration times',num_iterations)
    print('seed',seed)
    print('jitter value',jitter_value)
    mean_train_test_X, std_train_test_X, mean_p_x, std_p_x, var_p_x = prior_x_mean_var(prior_x_mean, prior_x_var,
                                                                                       N_Nstar, N, Q)
    # generate input X and observaation Y
    Y, samples_train_test_X, trainY_mode = inputX_observationY(output_scale_sqr_generate,
                                                               length_scale_inv_generate_1,
                                                               length_scale_inv_generate_2,
                                                               noise_inv_generate,
                                                               mean_train_test_X,
                                                               std_train_test_X,
                                                               D, N, Q, seed)
    print('samples_train_test_X',samples_train_test_X)

    print('Y',Y)

    # var and mean of q(X)
    # mean of q(X) mode:  "fixed", "pca-package", "pca-customize", "input", "random"
    mode = "input"
    mean_q_x, mean_q_x_save, log_sigma_q_x = q_X_var_mean(samples_train_test_X, Y, Q, N, seed, mode)
    # model hyperparameter
    output_scale_layer1 = torch.nn.Parameter(torch.tensor([output_scale_model], dtype=torch.float64).to(device),
                                             requires_grad=True)
    length_scale_inv_layer1 = torch.nn.Parameter(
        torch.tensor([length_scale_inv_model_1, length_scale_inv_model_2], dtype=torch.float64).to(device),
        requires_grad=True)
    noise_std_inv = torch.nn.Parameter(torch.tensor([noise_inv_model], dtype=torch.float64).to(device),
                                       requires_grad=True)
    print('initialize Y output scale',output_scale_layer1)
    print('initialize Y length scale', length_scale_inv_layer1)
    print('initialize Y noise beta',noise_std_inv)
    jitter_param = torch.tensor([jitter_value]).double().to(device)
    # generate inducing points
    Z = inducing_generate_Z(samples_train_test_X, M, seed, mode="sequential")
    print('input X size',samples_train_test_X.shape)
    print('Z size',Z.shape)
    print('Y',Y.shape)

    optimizer = optim.Adam([length_scale_inv_layer1, output_scale_layer1, noise_std_inv, mean_q_x, log_sigma_q_x],
                           lr=learning_rate)
    max_grad_norm = 0.5
    # num_iterations = 200
    loss_set = torch.empty(0).to(device)
    length_1_scale_set = torch.empty(0).to(device)
    length_2_scale_set = torch.empty(0).to(device)
    output_scale_set = torch.empty(0).to(device)
    noise_variance_set = torch.empty(0).to(device)
    mean_q_x_00_set = torch.empty(0).to(device)
    mean_q_x_01_set = torch.empty(0).to(device)
    var_q_x_00_set = torch.empty(0).to(device)
    var_q_x_01_set = torch.empty(0).to(device)
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = -loss_variational_bound(N, M, Q, D, Y, Z,
                                       length_scale_inv_layer1, output_scale_layer1, noise_std_inv,
                                       mean_p_x, mean_q_x,
                                       var_p_x, log_sigma_q_x,
                                       jitter_param
                                       )

        print('Iter %d/%d - Loss: %.6f  lengthscale: [%.6f//%.6f]  outputscale: %.6f  noise: %.6f ' % (
            i + 1, num_iterations, loss.item(),
            # length_scale_layer1.item(),
            length_scale_inv_layer1[0].item(),
            length_scale_inv_layer1[1].item(),
            output_scale_layer1.item(),
            noise_std_inv.item(),
            # jitter_param.item()
        ))

        # collect parameter data
        loss_set = torch.cat((loss_set, torch.tensor([loss.item()])))
        length_1_scale_set = torch.cat(
            (length_1_scale_set, torch.tensor([length_scale_inv_layer1[0].item()]).to(device)))
        length_2_scale_set = torch.cat(
            (length_2_scale_set, torch.tensor([length_scale_inv_layer1[1].item()]).to(device)))
        output_scale_set = torch.cat((output_scale_set, torch.tensor([output_scale_layer1.item()]).to(device)))
        noise_variance_set = torch.cat((noise_variance_set, torch.tensor([noise_std_inv.item()]).to(device)))
        # mean and var of q(x) 00, 01 collection
        # print('mean_q_x[0, 0].item()',mean_q_x[0, 0].item())
        mean_q_x_00_set = torch.cat((mean_q_x_00_set, torch.tensor([mean_q_x[0, 0].item()]).to(device)))
        mean_q_x_01_set = torch.cat((mean_q_x_01_set, torch.tensor([mean_q_x[0, 1].item()]).to(device)))
        var_q_x_00_set = torch.cat((var_q_x_00_set, torch.tensor([log_sigma_q_x[0, 0].item()]).to(device)))
        var_q_x_01_set = torch.cat((var_q_x_01_set, torch.tensor([log_sigma_q_x[0, 1].item()]).to(device)))

        loss.backward()
        # torch.nn.utils.clip_grad_norm_([output_scale_layer1, length_scale_layer1, noise_variance],
        #                                max_grad_norm)
        optimizer.step()
        print('Iter %d/%d - Loss: %.6f  lengthscale: [%.6f//%.6f]  outputscale: %.6f  noise: %.6f ' % (
            i + 1, num_iterations, loss.item(),
            # length_scale_layer1.item(),
            length_scale_inv_layer1[0].item(),
            length_scale_inv_layer1[1].item(),
            output_scale_layer1.item(),
            noise_std_inv.item(),
            # jitter_param.item()
        ))
        length_scale_inv_layer1.data = torch.clamp(length_scale_inv_layer1.data, min=1e-10, max=1e15)
        length_scale_inv_layer1[0].data = torch.clamp(length_scale_inv_layer1[0].data, min=1e-10, max=1e15)
        length_scale_inv_layer1[1].data = torch.clamp(length_scale_inv_layer1[1].data, min=1e-10, max=1e15)
        output_scale_layer1.data = torch.clamp(output_scale_layer1.data, min=1e-10, max=1e15)
        noise_std_inv.data = torch.clamp(noise_std_inv.data, min=1e-10, max=1e15)
        log_sigma_q_x.data = torch.clamp(log_sigma_q_x.data, min=1e-15, max=1e15)


    print('       - mean_q_x ', mean_q_x.detach())
    print('       - log_sigma_q_x', log_sigma_q_x.detach())
    std_q_x = torch.nn.functional.softplus(log_sigma_q_x)
    var_q_x = torch.square(std_q_x)
    print('       - var_q_x', var_q_x)
    print('Y', Y)
    print('X', samples_train_test_X)
    print('Z', Z)
    print('X', samples_train_test_X.shape)
    print('Z', Z.shape)
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} secs")

    # prediction mean
    # mean = prediction_train_X(samples_train_test_X,
    #                                   length_scale_inv_layer1,
    #                                   output_scale_layer1,
    #                                   noise_std_inv,
    #                                   Y)
    # # nrmse
    # for task in range(0, D):
    #     train_rmse = torch.mean(
    #         torch.pow(mean[:, task] - Y[:, task], 2)).sqrt()
    #     max_y = torch.max(Y[:, task])
    #     min_y = torch.min(Y[:, task])
    #     nrmse = train_rmse / (max_y - min_y)
    #     print('. NRMSE: %e ' % nrmse)


    # save data
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")

    file_name = f'GPLVM_ard_input_X_inverse(data_{N}_Q_{Q}_D_{D})_Y{trainY_mode}_LS[{length_scale_inv_generate_1}_{length_scale_inv_generate_2}]_OP{output_scale_sqr_generate}_NS{noise_inv_generate}_IP{M}_qM{mode}_Iter{num_iterations}_LearnR{learning_rate}_T{formatted_datetime}.pt'
    torch.save({'loss_set': loss_set,
                'length_1_scale_set': length_1_scale_set,
                'length_2_scale_set': length_2_scale_set,
                'out_scale_set': output_scale_set,
                'noise_variace_set': noise_variance_set,
                'input_X': samples_train_test_X,
                'Y': Y, 'Z': Z,
                'mean q_x mode':mode,
                'mean_q_x': mean_q_x.detach(),
                'var_q_x': log_sigma_q_x.detach(),
                'mean_q_x_00_set': mean_q_x_00_set,
                'mean_q_x_01_set': mean_q_x_01_set,
                'var_q_x_00_set': var_q_x_00_set,
                'var_q_x_01_set': var_q_x_01_set,
                '': mean_q_x_01_set,
                'pca_q_x': mean_q_x_save,
                'length_scale_inv_generate_1': length_scale_inv_generate_1,
                'length_scale_inv_generate_2': length_scale_inv_generate_2,
                'output_scale_generate': output_scale_sqr_generate,
                'noise_std_inv_generate': noise_inv_generate,
                'D': D, 'N': N, 'Q': Q, 'M': M,
                'time elapsed':(end_time - start_time)}, file_name)
    print(
        f'GPLVM_ard_inverse(data_{N}_Q_{Q}_D_{D})_KMeans_LS[{length_scale_inv_generate_1}_{length_scale_inv_generate_2}]_OP{output_scale_sqr_generate}_NS{noise_inv_generate}_IP{M}_Iter{num_iterations}_LearningR{learning_rate}')
    X_plot = np.arange(1, num_iterations + 1)
    Y1_f_x = np.array(loss_set)
    plt.plot(X_plot, Y1_f_x,
             label=f'loss: inv ls=[{length_scale_inv_layer1[0].item()}, {length_scale_inv_layer1[1].item()}]; os={output_scale_layer1.item()}; inv ns std={noise_std_inv.item()}')
    plt.title(
        f" GPLVM_ard_input_X_(data_{N}_Q_{Q}_D_{D})_KMeans_LS[{length_scale_inv_generate_1}_{length_scale_inv_generate_2}]_OP{output_scale_sqr_generate}_IP{M}_Iter{num_iterations}_LearningR{learning_rate} ")
    plt.xlabel('iteration time k')
    plt.ylabel('loss')
    plt.savefig(
        f'GPLVM_ard_input_X_inverse(data_{N}_Q_{Q}_D_{D})_KMeans_LS[{length_scale_inv_generate_1}_{length_scale_inv_generate_2}]_OP{output_scale_sqr_generate}_IP{M}_Iter{num_iterations}_LearnR{learning_rate}_T{formatted_datetime}.png')
    plt.show()


# parameters
torch.manual_seed(42)

N_star = 0
N = data_points
N_Nstar = N_star + N
Q = 2
D = 1
M = 100
seed = 42

prior_x_mean = 0.
prior_x_var = 1.
jitter_value = 0.00000001

main_train(N_Nstar, N, N_star, Q, D, M,
           prior_x_mean, prior_x_var, jitter_value,
           output_scale_sqr_generate,
           length_scale_inv_generate_1,
           length_scale_inv_generate_2,
           noise_inv_generate,
           output_scale_model,
           length_scale_inv_model_1,
           length_scale_inv_model_2,
           noise_inv_model,
           seed,
           learning_rate,
           num_iterations)


