// cholesky decomposition
// reuses the errorJ vector

/*
 * CUDA device side object that uses purely register memory to perform weighting-least-squares
 */

#ifndef CHOLESKY_CURVATURE_
#define CHOLESKY_CURVATURE_


#define CUDA_DEVICE_HOST __device__ __host__

class WLS {

	float r0c0,r1c0,r2c0,r3c0,r4c0,r5c0,r1c1,r2c1,r3c1,r4c1,r5c1,r2c2,r3c2,r4c2,r5c2,r3c3,r4c3,r5c3,r4c4,r5c4,r5c5; // makes it the same as declaring a right triangle matrix (this is a left triangle matrix)
public:

	float x0,x1,x2,x3,x4,x5;

	CUDA_DEVICE_HOST inline void add_mJ(const float m, float* J, float weight = 1.0f)
	{
		// compute weighted jacobian values
		float Jw0 = weight*J[0];
		float Jw1 = weight*J[1];
		float Jw2 = weight*J[2];
		float Jw3 = weight*J[3];
		float Jw4 = weight*J[4];
		float Jw5 = weight*J[5];

		r0c0 += Jw0*J[0];
		r1c0 += Jw0*J[1];
		r2c0 += Jw0*J[2];
		r3c0 += Jw0*J[3];
		r4c0 += Jw0*J[4];
		r5c0 += Jw0*J[5];

		r1c1 += Jw1*J[1];
		r2c1 += Jw1*J[2];
		r3c1 += Jw1*J[3];
		r4c1 += Jw1*J[4];
		r5c1 += Jw1*J[5];

		r2c2 += Jw2*J[2];
		r3c2 += Jw2*J[3];
		r4c2 += Jw2*J[4];
		r5c2 += Jw2*J[5];

		r3c3 += Jw3*J[3];
		r4c3 += Jw3*J[4];
		r5c3 += Jw3*J[5];

		r4c4 += Jw4*J[4];
		r5c4 += Jw4*J[5];

		r5c5 += Jw5*J[5];

		x0 += m*Jw0;
		x1 += m*Jw1;
		x2 += m*Jw2;
		x3 += m*Jw3;
		x4 += m*Jw4;
		x5 += m*Jw5;
	}

	CUDA_DEVICE_HOST inline void add_mJ_LVM(const float m, float* J, float weight = 1.0f, float lambda=0.0f)
	{
		// compute weighted jacobian values
		float Jw0 = weight*J[0];
		float Jw1 = weight*J[1];
		float Jw2 = weight*J[2];
		float Jw3 = weight*J[3];
		float Jw4 = weight*J[4];
		float Jw5 = weight*J[5];

		r0c0 += Jw0*J[0];
		r1c0 += Jw0*J[1];
		r2c0 += Jw0*J[2];
		r3c0 += Jw0*J[3];
		r4c0 += Jw0*J[4];
		r5c0 += Jw0*J[5];

		r1c1 += Jw1*J[1];
		r2c1 += Jw1*J[2];
		r3c1 += Jw1*J[3];
		r4c1 += Jw1*J[4];
		r5c1 += Jw1*J[5];

		r2c2 += Jw2*J[2];
		r3c2 += Jw2*J[3];
		r4c2 += Jw2*J[4];
		r5c2 += Jw2*J[5];

		r3c3 += Jw3*J[3];
		r4c3 += Jw3*J[4];
		r5c3 += Jw3*J[5];

		r4c4 += Jw4*J[4];
		r5c4 += Jw4*J[5];

		r5c5 += Jw5*J[5];

		x0 += m*Jw0;
		x1 += m*Jw1;
		x2 += m*Jw2;
		x3 += m*Jw3;
		x4 += m*Jw4;
		x5 += m*Jw5;

		r0c0 += lambda*r0c0;
		r1c1 += lambda*r1c1;
		r2c2 += lambda*r2c2;
		r3c3 += lambda*r3c3;
		r4c4 += lambda*r4c4;
		r5c5 += lambda*r5c5;
	}

	CUDA_DEVICE_HOST inline void do_cholesky(){

		// zero col
		r1c0/=r0c0;
		r2c0/=r0c0;
		r3c0/=r0c0;
		r4c0/=r0c0;
		r5c0/=r0c0;

		// one col - top entry
		r1c1 -= r0c0*r1c0*r1c0;
		// remaining entries
		r2c1 -= r0c0*r1c0*r2c0; r2c1 /= r1c1;
		r3c1 -= r0c0*r1c0*r3c0; r3c1 /= r1c1;
		r4c1 -= r0c0*r1c0*r4c0; r4c1 /= r1c1;
		r5c1 -= r0c0*r1c0*r5c0; r5c1 /= r1c1;

		// two col - top entry
		r2c2 -= r0c0*r2c0*r2c0 + r1c1*r2c1*r2c1;
		// remaining entries
		r3c2 -= r0c0*r2c0*r3c0 + r1c1*r2c1*r3c1; r3c2 /= r2c2;
		r4c2 -= r0c0*r2c0*r4c0 + r1c1*r2c1*r4c1; r4c2 /= r2c2;
		r5c2 -= r0c0*r2c0*r5c0 + r1c1*r2c1*r5c1; r5c2 /= r2c2;

		// three col - top entry
		r3c3 -= r0c0*r3c0*r3c0 + r1c1*r3c1*r3c1 + r2c2*r3c2*r3c2;
		// remaining entries
		r4c3 -= r0c0*r3c0*r4c0 + r1c1*r3c1*r4c1 + r2c2*r3c2*r4c2; r4c3 /= r3c3;
		r5c3 -= r0c0*r3c0*r5c0 + r1c1*r3c1*r5c1 + r2c2*r3c2*r5c2; r5c3 /= r3c3;

		// four col - top entry
		r4c4 -= r0c0*r4c0*r4c0 + r1c1*r4c1*r4c1 + r2c2*r4c2*r4c2 + r3c3*r4c3*r4c3;
		// remaining entries
		r5c4 -= r0c0*r4c0*r5c0 + r1c1*r4c1*r5c1 + r2c2*r4c2*r5c2 + r3c3*r4c3*r5c3; r5c4 /= r4c4;

		// five col - only entry
		r5c5 -= r0c0*r5c0*r5c0 + r1c1*r5c1*r5c1 + r2c2*r5c2*r5c2 + r3c3*r5c3*r5c3 + r4c4*r5c4*r5c4;
	}

	CUDA_DEVICE_HOST inline void perform_backsub()
	{
		// Back-substitute through L.
		x1 -= (r1c0 * x0);
		x2 -= (r2c0 * x0);
		x2 -= (r2c1 * x1);
		x3 -= (r3c0 * x0);
		x3 -= (r3c1 * x1);
		x3 -= (r3c2 * x2);
		x4 -= (r4c0 * x0);
		x4 -= (r4c1 * x1);
		x4 -= (r4c2 * x2);
		x4 -= (r4c3 * x3);
		x5 -= (r5c0 * x0);
		x5 -= (r5c1 * x1);
		x5 -= (r5c2 * x2);
		x5 -= (r5c3 * x3);
		x5 -= (r5c4 * x4);

		// Back-substitute through diagonal.
		x0 /= r0c0;
		x1 /= r1c1;
		x2 /= r2c2;
		x3 /= r3c3;
		x4 /= r4c4;
		x5 /= r5c5;

		// Back-substitute through L.T.
		x4 -= (r5c4 * x5);
		x3 -= (r4c3 * x4);
		x3 -= (r5c3 * x5);
		x2 -= (r3c2 * x3);
		x2 -= (r4c2 * x4);
		x2 -= (r5c2 * x5);
		x1 -= (r2c1 * x2);
		x1 -= (r3c1 * x3);
		x1 -= (r4c1 * x4);
		x1 -= (r5c1 * x5);
		x0 -= (r1c0 * x1);
		x0 -= (r2c0 * x2);
		x0 -= (r3c0 * x3);
		x0 -= (r4c0 * x4);
		x0 -= (r5c0 * x5);
	}

	CUDA_DEVICE_HOST inline void compute()
	{
		do_cholesky();
		perform_backsub();
	}

	CUDA_DEVICE_HOST inline void init()
	{
		// Initialise Matrix
		r0c0=r1c0=r1c1=r2c0=r2c1=r2c2=r3c0=r3c1=r3c2=r3c3=r4c0=r4c1=r4c2=r4c3=r4c4=r5c0=r5c1=r5c2=r5c3=r5c4=r5c5=0;
		// Initialise Vector
		x0=x1=x2=x3=x4=x5=0;
	}

	CUDA_DEVICE_HOST WLS(bool init_WLS=false)
	{
		if(init_WLS)
			init();
	}

//	friend ostream& operator<<(ostream& os, const WLS& wls);

};


#endif
