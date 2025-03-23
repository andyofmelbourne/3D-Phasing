import numpy as np
# import h5py
import pyopencl as cl


class D6_opencl():
    """
    Apply pixel perfect symmetry (P222)

    Only 2 P2 operations required

    assume fftshifted arrays with numpy fourier indexing

    qx = (i + N//2) % N - N//2
    qy = (j + N//2) % N - N//2
    qz = (k + N//2) % N - N//2

    N even
    2-fold z: x --> -x, y --> -y
    qx' = -qx
    (i' + N/2) % N - N/2 = - (i + N/2) % N + N/2
    (i' + N/2) % N = - (i + N/2) % N + N
    (i' + N/2) = - (i + N/2) % N + N + n N
    i' = - (i + N/2) % N + N/2 + n N
    i' = - (i + N/2) % N + N (n + 1/2)
    i' = [- (i + N/2) % N + N (n + 1/2)] % N

    i = [0, N-1]
    i'(i=0) = [- (N/2) % N + N (n + 1/2) ] % N
            = [n N] % N
            = 0 (n=0)

    i'(i=1) = [- N/2 - 1 + N (n + 1/2) ] % N
            = [n N - 1] % N
            = 0 (n=0)

    i'(i=N-1) = [- (N-1+N/2) % N + N (n + 1/2)] % N
              = [- (N/2-1) + N (n + 1/2)] % N
              = [1 + N n] % N
              = 1 (n=0)

    i'(i=N/2) = [- (N/2 + N/2) % N + N (n + 1/2)] % N
              = [N (n + 1/2)] % N
              = N/2 (n=0)
    """
    def __init__(self, shape, context, queue):
        self.shape = shape
        self.dtype = np.complex64
        self.temp1 = cl.array.empty(queue, shape, dtype=np.complex64)
        self.queue = queue
        self.context = context

        self.code = """
        #include <pyopencl-complex.h>

        // add
        __kernel void P2z(
            global cfloat_t *in,
            global cfloat_t *out
        )
        {
            int k   = get_global_id(0);
            int j   = get_global_id(1);
            int i   = get_global_id(2);
            int N   = get_global_size(0);

            int i2 = (- ((i + N/2) % N) + N/2 + N) % N;
            int j2 = (- ((j + N/2) % N) + N/2 + N) % N;

            int i_in = N*N*i + N*j + k;
            int i_out = N*N*i2 + N*j2 + k;

            out[i_out].x += in[i_in].x;
            out[i_out].y += in[i_in].y;
        }

        // add
        __kernel void P2x(
            global cfloat_t *in,
            global cfloat_t *out
        )
        {
            int k   = get_global_id(0);
            int j   = get_global_id(1);
            int i   = get_global_id(2);
            int N   = get_global_size(0);

            int j2 = (- ((j + N/2) % N) + N/2 + N) % N;
            int k2 = (- ((k + N/2) % N) + N/2 + N) % N;

            int i_in = N*N*i + N*j + k;
            int i_out = N*N*i + N*j2 + k2;

            out[i_out].x += in[i_in].x;
            out[i_out].y += in[i_in].y;
        }

        // assignment
        __kernel void P1(
            global cfloat_t *in,
            global cfloat_t *out
        )
        {
            int k   = get_global_id(0);
            int j   = get_global_id(1);
            int i   = get_global_id(2);
            int N   = get_global_size(0);

            int i_in = N*N*i + N*j + k;

            out[i_in].x = in[i_in].x;
            out[i_in].y = in[i_in].y;
        }

        // in place
        __kernel void divide(
            global cfloat_t *in,
            const float x
        )
        {
            int k   = get_global_id(0);
            int j   = get_global_id(1);
            int i   = get_global_id(2);
            int N   = get_global_size(0);

            int i_in = N*N*i + N*j + k;

            in[i_in].x /= x;
            in[i_in].y /= x;
        }

        // in place
        __kernel void fill(
            global cfloat_t *in,
            const float x
        )
        {
            int k   = get_global_id(0);
            int j   = get_global_id(1);
            int i   = get_global_id(2);
            int N   = get_global_size(0);

            int i_in = N*N*i + N*j + k;

            in[i_in].x = x;
            in[i_in].y = 0;
        }
        """

        self.code_cl = cl.Program(context, self.code).build()
        self.P1 = self.code_cl.P1
        self.P2z = self.code_cl.P2z
        self.P2x = self.code_cl.P2x
        self.divide = self.code_cl.divide
        self.fill = self.code_cl.fill

    def apply(self, array_cl):
        """
        inplace opperation
        """
        # temp1[:] = array_cl
        self.P1(
            self.queue,
            self.shape,
            None,
            array_cl.data,
            self.temp1.data
        )

        # temp1 += array_cl[P2z[r]]
        # temp1 = P1 + P2z
        self.P2z(
            self.queue,
            self.shape,
            None,
            array_cl.data,
            self.temp1.data,
        )

        # array_cl[:] = temp1
        # array_cl = P1 + P2z
        self.P1(
            self.queue,
            self.shape,
            None,
            self.temp1.data,
            array_cl.data
        )

        # array_cl += temp1[P2x[r]]
        # array_cl = P1 + P2z + P2x + P2z . P2x
        self.P2x(
            self.queue,
            self.shape,
            None,
            self.temp1.data,
            array_cl.data,
        )

        # array_cl /= 4
        self.divide(
            self.queue,
            self.shape,
            None,
            array_cl.data,
            np.float32(4.)
        )
        return array_cl
