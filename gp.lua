require 'gnuplot'
require 'sys'

function sqexp(l, a, b)
  local d = torch.dist(a, b)
  return torch.exp(-d^2 / (2*l))
end

function nexp(l, a, b)
  local d = torch.dist(a, b)
  return torch.exp(-d / l)
end

function pseudo_inv(M, eps)
  local u,s,v = torch.svd(M, 'S')
  for i=1,s:size(1) do
    if torch.abs(s[i]) > eps then
      s[i] = 1/s[i]
    else
      s[i] = 0
    end
  end

  return v*torch.diag(s)*u:t()
end

function outer(f, x, y)
  local n = x:size(1)
  local k = y:size(1)
  local M = torch.Tensor(n, k)

  for i=1,n do
    for j=1,k do
      M[{i, j}] = f(x[i], y[j])
    end
  end

  return M
end

function krig(kern, X, Y, nu)
  if nu == nil then nu = 1e-3 end

  if X:dim() == 0 then
    return function(p)
      return 0, kern(0, 0)
    end
  end

  assert(X:dim() == 2)
  assert(Y:dim() == 1)
  assert(X:size(1) == Y:size(1))

  local sig22  = outer(kern, X, X) + torch.eye(X:size(1))*nu
  --local sig22i = torch.inverse(sig22)
  local sig22i = pseudo_inv(sig22, 1e-6)

  return function(p)
    assert(p:dim() == 1)

    local Xp    = p:reshape(1, p:size(1))
    local sig11 = outer(kern, Xp, Xp)
    local sig12 = outer(kern, Xp, X)
    local sig21 = sig12:t()
    local mu    = sig12*(sig22i*Y)
    local var   = sig11 - sig12*(sig22i*sig21)

    return mu[1], var[{1, 1}]
  end
end

function gaussian2(p)
  local x = p[1]
  local y = p[2]

  local g1 = torch.exp(-((3.0*x+0.75)^2 + (3.0*y+0.75)^2))
  local g2 = torch.exp(-((3.0*x-0.75)^2 + (3.0*y-0.75)^2))

  return 0.6*g1 + 0.4*g2
end

obj = gaussian2
noise = 0.01
smooth = 0.2

function kern(a, b)
  return nexp(smooth, a, b)
end

function gain(mu, sigma, ts)
  local b = 5.9
  local t = (ts - mu) / sigma
  return sigma*(torch.log(1 + b^-t)/torch.log(b))
end

function plot(X, Y)
  f = krig(kern, X, Y, noise)

  x = torch.linspace(-1, 1, 32)
  y = x

  local max = Y:max()

  z  = torch.Tensor(x:size(1), x:size(1))
  zc = torch.Tensor(x:size(1), x:size(1))
  g  = torch.Tensor(x:size(1), x:size(1))

  local data = {'\n'}

  for i=1,x:size(1) do
    for j=1,y:size(1) do
      local p = torch.Tensor{x[i], y[j]}
      local mu, sigma = f(p)
      z[{i, j}]  = mu
      zc[{i, j}] = sigma
      g[{i, j}]  = gain(mu, sigma, max)
    end
  end

  for i=1,x:size(1) do
    for j=1,y:size(1) do
      table.insert(data, string.format('%g %g %g %g\n', x[i], y[j], z[{i,j}], zc[{i,j}]))
    end
    table.insert(data, '\n')
  end
  table.insert(data, 'e\n')

  for i=1,X:size(1) do
    local p = X[i]
    table.insert(data, string.format('%g %g %g\n', p[1], p[2], Y[i]))
  end
  table.insert(data, 'e\n')

  gnuplot.raw('set contour base')
  gnuplot.raw('set cntrparam bspline')
  gnuplot.raw('set cntrparam levels auto')
  gnuplot.raw('unset hidden3d')
  gnuplot.raw('set zrange  [-0.25:0.8]')
  gnuplot.raw('set cbrange [0:1]')
  -- gnuplot.raw('set palette')
  gnuplot.raw('set palette rgb 30,31,32')
  gnuplot.raw('splot "-" with lines palette, "-" with points lt 3 ps 4')
  gnuplot.raw(table.concat(data))

  local cur_max = -1, max_i, max_j
  for i=1,x:size(1) do
    for j=1,y:size(1) do
      local gij = g[{i, j}]
      if gij > cur_max then
        cur_max = gij
        max_i = i
        max_j = j
      end
    end
  end

  return torch.Tensor({x[max_i], y[max_j]})
end

--X = torch.Tensor{{-2,-2}}

X = torch.rand(1, 2)*2 - 1
Y = torch.Tensor(X:size(1))
for i=1,Y:size(1) do
  Y[i] = obj(X[i])
end

function cycle()
  p = plot(X, Y)
  X = X:cat(p:reshape(1,2),1)
  y = obj(p) + torch.randn(1)[1]*noise
  Y = Y:cat(torch.Tensor{y})
end

function loop()
  for i=1,16 do
    cycle()
    sys.sleep(0.5)
  end
end

function tested()
  gnuplot.plot({X, '+'})
end

-- o = torch.Tensor({0,0})
-- x = torch.linspace(-1,1)
-- --z = outer(function(x, y) return sqexp(0.2, o, torch.Tensor({x, y})) end, x, x)
-- z = outer(function(x, y) return gaussian2(torch.Tensor({x, y})) end, x, x)
-- gnuplot.splot(z)
