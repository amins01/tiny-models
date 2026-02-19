export function generateGaussianNoise(size: number) {
    const data = new Float32Array(size);
    for (let i = 0; i < size; i += 2) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        const mag = Math.sqrt(-2.0 * Math.log(u));
        const z0 = mag * Math.cos(2.0 * Math.PI * v);
        const z1 = mag * Math.sin(2.0 * Math.PI * v);
        data[i] = z0;
        if (i + 1 < size) {
            data[i + 1] = z1;
        }
    }
    return data;
}