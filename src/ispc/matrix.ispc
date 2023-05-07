
export uniform float to_float(uniform const int32 n)
{
    return n;
}

export uniform float vector_dot(
    uniform const int32 size,
    float uniform const * uniform const v0,
    float uniform const * uniform const v1)
{
    float t = 0.0f;
    foreach (i = 0 ... size){
        t = v0[i]*v1[i];
    }
    return reduce_add(t);
}

export void vector_dot_batch(
    float uniform * uniform r,
    uniform uint32 batch_size,
    uniform uint32 size,
    float uniform const * uniform const v0,
    float uniform const * uniform const v1)
{
    for(uniform uint32 i=0; i<batch_size; ++i){
        float uniform const * uniform const tv0 = v0 + i*size;
        float uniform const * uniform const tv1 = v1 + i*size;
        float t = 0.0f;
        foreach (j = 0 ... size){
            t = tv0[j]*tv1[j];
        }
        r[i] = reduce_add(t);
    }
}

export void matrix_mul(
    float uniform * uniform r,
    uniform const int32 r0,
    uniform const int32 c0,
    uniform const int32 c1,
    float uniform const * uniform const m0,
    float uniform const * uniform const m1)
{
    for(uniform int32 i = 0; i < r0; ++i) {
        for(uniform int32 j = 0; j < c1; ++j) {
            float t = 0.0f;
            foreach(k=0 ... c0){
                t = m0[i*c0+k] * m1[k*c1+j];
            }
            r[i*c1+j] = reduce_add(t);
        }
    }
}

export void matrix_add(
    float uniform * uniform r,
    uniform const int32 size,
    float uniform const * uniform const m0,
    float uniform const * uniform const m1)
{
    foreach(i=0 ... size){
        r[i] = m0[i] + m1[i];
    }
}

export void matrix_muladd(
    float uniform * uniform r,
    uniform const int32 r0,
    uniform const int32 c0,
    uniform const int32 c1,
    float uniform const * uniform const m0,
    float uniform const * uniform const m1,
    float uniform const * uniform const m2)
{
    for(uniform int32 i = 0; i < r0; ++i) {
        for(uniform int32 j = 0; j < c1; ++j) {
            float t = 0.0f;
            foreach(k=0 ... c0){
                t = m0[i*c0+k] * m1[k*c1+j];
            }
            r[i*c1+j] = reduce_add(t) + m2[i*c1+j];
        }
    }
}

export void step(uniform int32 size, float uniform * uniform r, float uniform const * uniform const x)
{
    for(uniform int32 i=0; i<size; ++i){
        r[i] = 0.0f<x[i]? 1.0f : 0.0f;
    }
}

export uniform float max_p(uniform int32 size, float uniform const * uniform const x)
{
    float t = 0.0f;
    foreach(i=0 ... size){
        t = x[i];
    }
    return reduce_max(t);
}

export uniform float min_p(uniform int32 size, float uniform const * uniform const x)
{
    float t = 0.0f;
    foreach(i=0 ... size){
        t = x[i];
    }
    return reduce_min(t);
}

export uniform float sum(uniform int32 size, float uniform const * uniform const x)
{
    float t = 0.0f;
    foreach(i=0 ... size){
        t = x[i];
    }
    return reduce_add(t);
}

export void sigmoid(uniform int32 size, float uniform * uniform r, float uniform const * uniform const x)
{
    for(uniform int32 i=0; i<size; ++i){
        r[i] = 1.0f/(1.0f+exp(-x[i]));
    }
}

export void relu(uniform int32 size, float uniform * uniform r, float uniform const * uniform const x)
{
    for(uniform int32 i=0; i<size; ++i){
        r[i] = 0.0f<=x[i]? x[i] : 0.0f;
    }
}

export void softmax(uniform int32 size, float uniform * uniform r, float uniform const * uniform const x)
{
    float c = max_p(size, x);
    foreach(i=0 ... size){
        r[i] = exp(x[i]-c);
    }
    uniform float sum = max(sum(size, r), 1.0e-6f);
    uniform float inv = 1.0f/sum;
    foreach(i=0 ... size){
        r[i] *= inv;
    }
}

export void array_to_float(uniform uint32 size, float uniform * uniform r, uint8 uniform const * uniform const x, uniform float scale)
{
    foreach(i=0 ... size){
        r[i] = scale*x[i];
    }
}

