// calculate acc.s0+value with error compensation
// see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
static inline float2 kahan_sum(float2 acc, float value)
{
    if (value)
    {
        float y = value - acc.s1;
        float t = acc.s0 + y;
        float err = (t - acc.s0) - y;
        return (float2)(t, err);
    }
    else
    {
        return acc;
    }
}

// calculate a + b with error compensation
static inline float2 compensated_sum(float2 a, float2 b)
{
    float sum, err = a.s1 + b.s1;

    if (fabs(a.s0) > fabs(b.s0))
    {
        float y = b.s0 - err;
        sum = a.s0 + y;
        err = (sum - a.s0) - y;
    }
    else
    {
        float y = a.s0 - err;
        sum = b.s0 + y;
        err = (sum - b.s0) - y;
    }
    return (float2)(sum, err);
}
