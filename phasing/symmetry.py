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


class D6_image_cl():
    """
    Use an opencl image to apply non-pixel perfect
    symmetry to array in Fourier space

    assume fftshifted arrays
    assume complex arrays
    apply to phase

    i' = fftshifted indices
    i  = linear pixel index
    i' = ifftshift(i)
    [4, 5, 6, 7, 0, 1, 2, 3] = ifftshift([0, 1, 2, 3, 4, 5, 6, 7])

    For even arrays shapes: fftshift = ifftshift
    ifftshift(i, N):
        i' = (N//2 + i) % N

    fftshift(i, N):
        i = (N//2 + i') % N
    """

    def __init__(self, shape, context, queue):
        mf = cl.mem_flags

        # copy I as an opencl "image" for trilinear sampling
        image_format = cl.ImageFormat(
            cl.channel_order.R,
            cl.channel_type.FLOAT
        )
        flags = mf.READ_WRITE

        t = np.empty(shape, dtype=np.float32)
        self.I_cl = cl.Image(context, flags, image_format, shape=shape[::-1])
        # self.I_cl = cl.create_image(context, flags, image_format, shape=shape[::-1])
        self.amp = cl.Buffer(context, mf.READ_WRITE, t.nbytes)
        self.phase = cl.Buffer(context, mf.READ_WRITE, t.nbytes)

        # cl.enqueue_copy(
        #     queue,
        #     I_cl,
        #     np.ascontiguousarray(ar.T.astype(np.float32)),
        #     is_blocking=True,
        #     origin=(0, 0, 0),
        #     region=ar.shape[::-1]
        # )
        self.code = """
        #pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
        constant sampler_t interpolation =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST ;

        // fill image with fftshifted array
        __kernel void amp_phase (
            global float2 *in,
            global float *amp,
            global float *phase
        ){
        int i   = get_global_id(0);

        amp[i] = length(in[i]);
        phase[i] = atan2(in[i].y, in[i].x);
        }

        __kernel void amp_phase_inv (
            global float2 *in,
            global float *amp,
            global float *phase
        ){
        int i   = get_global_id(0);

        in[i].x = amp[i] * cos(phase[i]);
        in[i].y = amp[i] * sin(phase[i]);
        }

        __kernel void real_imag (
            global float2 *in,
            global float *real,
            global float *imag
        ){
        int i   = get_global_id(0);

        real[i] = in[i].x;
        imag[i] = in[i].y;
        }

        __kernel void real_imag_inv (
            global float2 *in,
            global float *real,
            global float *imag
        ){
        int i   = get_global_id(0);

        in[i].x = real[i];
        in[i].y = imag[i];
        }

        __kernel void fill (
            __write_only image3d_t I,
            global float *array
        ){
        int k   = get_global_id(0);
        int j   = get_global_id(1);
        int i   = get_global_id(2);
        int N   = get_global_size(0);

        int4 n;
        n.x = (i + N/2) % N;
        n.y = (j + N/2) % N;
        n.z = (k + N/2) % N;
        n.w = 0;

        int l = N*N*i + N*j + k;

        float4 v;
        v.x = array[l];
        v.y = 0.;
        v.z = 0.;
        v.w = 0.;

        write_imagef(I, n, v);
        }

        // apply symmetry
        __kernel void D6 (
            __read_only image3d_t I,
            global float *array
        ){
        int4 i;
        i.z = get_global_id(0);
        i.y = get_global_id(1);
        i.x = get_global_id(2);
        int N   = get_global_size(0);

        // ravelled index
        int l = N*N*i.x + N*i.y + i.z;

        // location of origin in unshifted array
        float i0 = (float)(N/2);

        // r-coordintes in I space
        // r = ifftshift(i) - i0
        float4 r;
        r.x = (float)((i.x + N/2) % N) - i0;
        r.y = (float)((i.y + N/2) % N) - i0;
        r.z = (float)((i.z + N/2) % N) - i0;
        r.w = 0;

        // indices in I space
        // n = x + i0 + 0.5
        float4 n;
        n.w = 0;

        // temp
        float x, y, z;

        // output value
        float v = 0.;

        float c = 0.5;
        float s = 0.8660254037844386;

        int ii, jj;
        for (ii=0; ii<2; ii++){
            // 2-fold about x
            r.y = -r.y;
            r.z = -r.z;

            // 6x pi/3 rotation about z
            // x' = x * c - y * s;
            // y' = x * s + y * c;
            for (jj=0; jj<6; jj++){
                x = r.x * c - r.y * s;
                y = r.x * s + r.y * c;
                r.x = x;
                r.y = y;

                n.x = x + i0 + 0.5;
                n.y = y + i0 + 0.5;
                n.z = r.z + i0 + 0.5;

                v += read_imagef(I, interpolation, n).x;
            }
        }
        v /= 12;

        array[l] = v;
        }
        """
        self.code_cl = cl.Program(context, self.code).build()
        self.fill_image = self.code_cl.fill
        self.D6 = self.code_cl.D6
        self.amp_phase = self.code_cl.amp_phase
        self.amp_phase_inv = self.code_cl.amp_phase_inv
        self.real_imag = self.code_cl.real_imag
        self.real_imag_inv = self.code_cl.real_imag_inv
        self.context = context
        self.queue = queue
        self.shape = shape
        self.size = np.prod(shape)

    def fill(self, ar_cl):
        self.fill_image(
            self.queue,
            self.shape,
            None,
            self.I_cl,
            ar_cl,
        )

    def apply_real(self, ar_cl):
        # send fftshifted ar_cl to un-fftshifted image
        self.fill(ar_cl)

        # apply symmetry operators
        self.D6(
            self.queue,
            self.shape,
            None,
            self.I_cl,
            ar_cl,
        )

    def apply(self, ar_cl):
        self.real_imag(
            self.queue,
            (self.size,),
            None,
            ar_cl.data,
            self.amp,
            self.phase,
        )

        # just apply to phase
        self.apply_real(self.amp)
        self.apply_real(self.phase)

        self.real_imag_inv(
            self.queue,
            (self.size,),
            None,
            ar_cl.data,
            self.amp,
            self.phase,
        )
