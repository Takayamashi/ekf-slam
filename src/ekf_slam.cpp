#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <cmath>
#include <tuple>
#include <random>
#include <Eigen/LU>
#include <unsupported/Eigen/MatrixFunctions>

#define deg_to_rad(deg) (deg*M_PI) / 180.
#define rad_to_deg(rad) (rad*180) / M_PI

double dt = 0.1; // time tick [s]
double SIM_TIME = 50.0; // simulation time [s]
double MAX_RANGE = 15.0; // maximum observation range [m]
double W_MIN = 0.01; // minimum angular velocity [rad/s]
double M_DIST_TH = 2.0; // Threshold of Mahalanobis distance for data association
const int STATE_SIZE = 3; // State size [x,y,theta]
const int LM_SIZE = 2; // LM srate size [x,y]

// Simulation parameter
std::vector<double> Qsim = {pow(0.2, 2), pow(deg_to_rad(1.0), 2)};
std::vector<double> Rsim = {pow(1.0, 2), pow(deg_to_rad(10.0), 2)};

// EKF state covariance
Eigen::Vector3d cx_diag(pow(0.5, 2), pow(0.5, 2), pow(deg_to_rad(30.0), 2));
Eigen::Matrix<double, 3, 3> Cx = cx_diag.asDiagonal();


double pi_2_pi(double angle)
{
    double z_angle = fmod(angle + M_PI, 2.0 * M_PI) - M_PI;
    return z_angle;
}


Eigen::Matrix<double, 2, 1> calc_input()
{
    Eigen::Matrix<double, 2, 1> u;
    double v = 3.0; // [m/s]
    double yawrate = 0.0; // [rad/s]
    u << v,
         yawrate;
    return u;
}


Eigen::Matrix<double, 3, 1> motion_model(Eigen::Matrix<double, 3, 1> x, Eigen::Matrix<double, 2, 1> u)
{
    Eigen::Matrix<double, 3, 1> x_;
    Eigen::Matrix<double, 3, 3> F;
    double theta = x(2, 0);
    double omega = u(1, 0);
    F << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0;

    if(std::fabs(omega) <= W_MIN)
    {
        Eigen::Matrix<double, 3, 2> B;
        B << dt * cos(theta), 0.0,
             dt * sin(theta), 0.0,
             0.0, dt;

        x_ = F * x + B * u;
    }
    else
    {
        double vw =  u(0, 0) / omega;
        Eigen::Matrix<double, 3, 1> Bu;
         Bu << - vw * sin(theta) + vw * sin(theta + omega * dt),
              vw * cos(theta) - vw * cos(theta + omega * dt),
              omega * dt;
        x_ = F * x + Bu;
    }

    return x_;
}


int calc_n_LM(Eigen::MatrixXd x)
{
    int n = int((x.rows() - STATE_SIZE) / LM_SIZE);
    return n;
}


std::tuple<Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>, Eigen::MatrixXd> jacob_motion(Eigen::Matrix<double, 3, 1> x, Eigen::Matrix<double, 2, 1> u)
{
    // Jacobian of Motion model
    Eigen::Matrix<double, 3, 3> jF;
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> G;
    Eigen::MatrixXd Fx(STATE_SIZE, STATE_SIZE + LM_SIZE * calc_n_LM(x));
    Fx << Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE), Eigen::MatrixXd::Zero(STATE_SIZE, LM_SIZE * calc_n_LM(x));
    double theta = x(2, 0);
    double v = u(0, 0);
    double w = u(1, 0);
    if(fabs(w) <= W_MIN)
    {
        jF << 0.0, 0.0, - dt * v * sin(theta),
              0.0, 0.0, dt * v * cos(theta),
              0.0, 0.0, 0.0;
    }
    else
    {
        double vw = v / w;
        jF << 0.0, 0.0, - vw * cos(theta) + vw * cos(theta + w*dt),
              0.0, 0.0, - vw * sin(theta) + vw * sin(theta + w*dt),
              0.0, 0.0, 0.0;
    }

    G = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE) + Fx.transpose() * jF * Fx;

    return std::forward_as_tuple(G, Fx);
}


Eigen::MatrixXd calc_LM_Pos(Eigen::MatrixXd x, Eigen::Matrix<double, 3, 1> z)
{
    Eigen::MatrixXd zp(2, 1);

    zp << x(0, 0) + z(0) * std::cos(x(2, 0) + z(1)),
          x(1, 0) + z(0) * std::sin(x(2, 0) + z(1));

    return zp;
}


Eigen::Matrix<double, LM_SIZE, 1> get_LM_Pos_from_state(Eigen::MatrixXd x, int ind)
{
    Eigen::Matrix<double, LM_SIZE, 1> lm;
    lm = x.block(STATE_SIZE + LM_SIZE * ind, 0, 2, 1);
    return lm;
}


Eigen::MatrixXd jacobH(double q, Eigen::Matrix<double, LM_SIZE, 1> delta, Eigen::MatrixXd x, int i)
{
    double sq = std::sqrt(q);
    Eigen::Matrix<double, 2, 5> G;
    int nLM;

    G << -sq * delta(0, 0) / q, -sq * delta(1, 0) / q, 0.0, sq * delta(0, 0) / q, sq * delta(1, 0) / q,
         delta(1, 0) / q, - delta(0, 0) / q, -1.0, - delta(1, 0) / q, delta(0, 0) / q;
    nLM = calc_n_LM(x);
    Eigen::MatrixXd F1(STATE_SIZE, STATE_SIZE + LM_SIZE * nLM);
    Eigen::MatrixXd F2(2, STATE_SIZE + LM_SIZE * nLM);
    Eigen::MatrixXd F(STATE_SIZE + 2, STATE_SIZE + LM_SIZE * nLM);
    F1 << Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE), Eigen::MatrixXd::Zero(3, 2 * nLM);
    F2 << Eigen::MatrixXd::Zero(2, 3), Eigen::MatrixXd::Zero(2, 2*(i-1)), Eigen::MatrixXd::Identity(2, 2), Eigen::MatrixXd::Zero(2, 2*nLM - 2*i);
    F << F1,
         F2;


    Eigen::MatrixXd H(2, STATE_SIZE + LM_SIZE * nLM);
    H = G * F;

    return H;
}


std::tuple<Eigen::Matrix<double, 2, 1>, Eigen::Matrix<double, 2, 2>, Eigen::MatrixXd> calc_innovation(
    Eigen::Matrix<double, LM_SIZE, 1> lm, Eigen::MatrixXd xEst, Eigen::MatrixXd PEst, Eigen::Matrix<double, 2, 1> z, int LMid)
{
    Eigen::Matrix<double, LM_SIZE, 1> delta;
    Eigen::Matrix<double, 1, 1> q_matrix;
    double q;
    double zangle;
    Eigen::Matrix<double, 2, 1> zp;
    Eigen::Matrix<double, 2, 1> y;
    Eigen::Matrix<double, 2, 2> S;
    Eigen::MatrixXd H;

    delta = lm - xEst.block(0, 0, 2, 1);
    q_matrix = delta.transpose() * delta;
    q = q_matrix(0, 0);
    zangle = std::atan2(delta(1, 0), delta(0, 0)) - xEst(2, 0);
    zp << std::sqrt(q),
          pi_2_pi(zangle);
    y = (z - zp);
    y(1, 0) = pi_2_pi(y(1, 0));
    H = jacobH(q, delta, xEst, LMid + 1);
    S = H * PEst * H.transpose() + Cx.block(0, 0, 2, 2);

    return std::forward_as_tuple(y, S, H);
}


int search_correspond_LM_ID(Eigen::MatrixXd xAug, Eigen::MatrixXd PAug, Eigen::Matrix<double, 2, 1> zi)
{
    int nLM = calc_n_LM(xAug);
    std::vector<double> mdist;
    Eigen::Matrix<double, LM_SIZE, 1> lm;

    for(int i=0; i < nLM; i++)
    {
        Eigen::Matrix<double, 2, 1> y;
        Eigen::Matrix<double, 2, 2> S;
        Eigen::MatrixXd H;
        lm = get_LM_Pos_from_state(xAug, i);
        std::tie(y, S, H) = calc_innovation(lm, xAug, PAug, zi, i);
        Eigen::Matrix<double, 1, 1> m_value = y.transpose() * S.inverse() * y;
        double m = m_value(0, 0);
        mdist.push_back(m);
    }

    mdist.push_back(M_DIST_TH);
    int minid = std::min_element(mdist.begin(),mdist.end()) - mdist.begin();

    return minid;
}


std::tuple< Eigen::Matrix<double, 3, 1>, std::vector<Eigen::Matrix<double, 3, 1>>, Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 2, 1> > observation(Eigen::Matrix<double, 3, 1> xTrue,
                    Eigen::Matrix<double, 3, 1> xd, Eigen::Matrix<double, 2, 1> u, std::vector<Eigen::Matrix<double, 1, 2>> RFID)
{
    Eigen::Matrix<double, 3, 1> xTrue_;
    xTrue_ = motion_model(xTrue, u);
    std::vector<Eigen::Matrix<double, 3, 1>> z;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> dist(0.0, 1.0);
    Eigen::Matrix<double, 3, 1> xd_;

    // observate landmark
    for(int i=0; i < RFID.size(); i++)
    {
        double dx = RFID[i](0, 0) - xTrue(0, 0);
        double dy = RFID[i](0, 1) - xTrue(1, 0);
        double r = std::sqrt(std::pow(dx, 2) + std::pow(dy, 2));
        double phi = std::atan2(dy, dx) - xTrue(2, 0);
        if(r <= MAX_RANGE)
        {
            double rn = r + dist(engine) * Qsim[0];
            double phin = phi + dist(engine) * Qsim[1];
            Eigen::Matrix<double, 3, 1> zn;
            zn << rn,
                  phin,
                  i;
            z.push_back(zn);
        }
    }

    Eigen::Matrix<double, 2, 1> ud;
    double u0 = u(0, 0) + dist(engine) * Rsim[0];
    double u1 = u(1, 0) + dist(engine) * Rsim[1];
    ud << u0,
          u1;

    xd_ = motion_model(xd, ud);

    return std::forward_as_tuple(xTrue_, z, xd_, ud);
}


std::tuple< Eigen::MatrixXd, Eigen::MatrixXd > ekf_slam(Eigen::MatrixXd xEst, Eigen::MatrixXd PEst,
        Eigen::Matrix<double, 2, 1> u, std::vector<Eigen::Matrix<double, 3, 1>> z)
{
    // Predict
    int S = STATE_SIZE;
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> G;
    Eigen::MatrixXd Fx;
    Eigen::Matrix<double, 2, 2> initP;

    xEst.block(0, 0, 3, 1) = motion_model(xEst.block(0, 0, 3, 1), u);
    std::tie(G, Fx) = jacob_motion(xEst.block(0, 0, 3, 1), u);
    PEst.block(0, 0, 3, 3) = G.transpose() * PEst.block(0, 0, 3, 3) * G + Fx.transpose() * Cx * Fx;
    initP = Eigen::MatrixXd::Identity(2, 2);

    // Update
    for(int iz=0; iz < z.size(); iz++) // for each observation
    {
        int minid = search_correspond_LM_ID(xEst, PEst, z[iz].block(0, 0, 2, 1));
        int nLM = calc_n_LM(xEst);
        if(minid == nLM)
        {
            // Extend state and covariance matrix
            Eigen::MatrixXd xAug;
            Eigen::MatrixXd PAug;
            Eigen::MatrixXd PAug_left;
            Eigen::MatrixXd PAug_right;
            xAug.resize(xEst.rows() + LM_SIZE, 1);
            PAug.resize(PEst.rows() + LM_SIZE, PEst.cols() + LM_SIZE);
            PAug_left.resize(PEst.rows() + LM_SIZE, PEst.cols());
            PAug_right.resize(PEst.rows() + LM_SIZE, LM_SIZE);
            Eigen::MatrixXd zp(2, 1);
            zp = calc_LM_Pos(xEst, z[iz]);
            xAug << xEst,
                    zp;


            PAug_left << PEst,
                         Eigen::MatrixXd::Zero(LM_SIZE, PEst.cols());


            PAug_right << Eigen::MatrixXd::Zero(PEst.rows(), LM_SIZE),
                          initP;


            PAug << PAug_left, PAug_right;


            xEst = xAug;
            PEst = PAug;
        }
        Eigen::Matrix<double, LM_SIZE, 1> lm;
        Eigen::Matrix<double, 2, 1> y;
        Eigen::Matrix<double, 2, 2> S;
        Eigen::MatrixXd H;
        Eigen::MatrixXd K;
        lm = get_LM_Pos_from_state(xEst, minid);
        std::tie(y, S, H) = calc_innovation(lm, xEst, PEst, z[iz].block(0, 0, 2, 1), minid);

        K = (PEst * H.transpose()) * S.inverse();
        xEst = xEst + (K * y);
        PEst = (Eigen::MatrixXd::Identity(xEst.rows(), xEst.rows()) - (K * H)) * PEst;
    }


    xEst(2, 0) = pi_2_pi(xEst(2, 0));


    return std::forward_as_tuple(xEst, PEst);
}


int main()
{
    // Landmarks Setting
    int RFIDN = 11;
    std::vector<Eigen::Matrix<double, 1, 2>> RFID;
    for (int i = 0; i < RFIDN; i++)
    {
        Eigen::Matrix<double, 1, 2> RFID_;
        // Straight Landmarks
        RFID_ << 5.0*i, 0.0;
        RFID.push_back(RFID_);
    }

    double time = 0.0;

    // Initial Setting
    Eigen::MatrixXd xEst;
    Eigen::Matrix<double, 3, 1> xTrue;
    Eigen::MatrixXd PEst;
    Eigen::Matrix<double, 3, 1> xDR;

    xEst.resize(3, 1);
    xEst << 0.0,
            -2.0,
            0.0;
    xTrue << 0.0,
             -2.0,
             0.0;
    PEst = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    xDR << 0.0,
           -2.0,
           0.0;

    Eigen::Matrix<double, 2, 1> u;
    std::vector<Eigen::Matrix<double, 3, 1>> z;
    Eigen::Matrix<double, 2, 1> ud;


    while (SIM_TIME >= time)
    {
        time += dt;
        u = calc_input();
        std::tie(xTrue, z, xDR, ud) = observation(xTrue, xDR, u, RFID);
        std::tie(xEst, PEst) = ekf_slam(xEst, PEst, ud, z);
        double x = xEst(0, 0);
        double y = xEst(1, 0);
        double theta = xEst(2, 0);
        std::cout << "--------------------" << std::endl;
        std::cout << x << std::endl;
        std::cout << y << std::endl;
        std::cout << theta << std::endl;
        std::cout << "--------------------" << std::endl;
    }


    return 0;
}
