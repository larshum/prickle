let softmax_row [N] (x : [N]f32) : [N]f32 =
  let m = reduce f32.max (-f32.inf) x in
  let t = map (\xi -> f32.exp (xi - m)) x in
  let s = reduce (+) 0.0 t in
  map (\xi -> xi / s) t

entry softmax [M][N] (x: [M][N]f32) : [M][N]f32 =
  map softmax_row x
