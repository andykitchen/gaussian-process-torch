require 'gnuplot'

function sqexp(l, a, b)
  return torch.exp(-torch.abs(b - a)^2 / (2*l))
end

function outer(f, x, y)
  assert(x:dim() == 1)
  assert(y:dim() == 1)

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

function gain(mu, sigma, ts)
  local b = 5.9
  local t = (ts - mu) / sigma
  return sigma*(torch.log(1 + b^-t)/torch.log(b))
end

function krig(kern, X, Y, nu)
  if nu == nil then nu = 1e-3 end

  assert(X:dim() == 1)
  assert(Y:dim() == 1)

  assert(X:isSameSizeAs(Y))

  if X:size(1) == 0 then
    return 0, kern(0, 0)
  end

  local sig22  = outer(kern, X, X) + torch.eye(X:size(1))*nu
  --local sig22i = torch.inverse(sig22)
  local sig22i = pseudo_inv(sig22, 1e-6)

  return function(p)
    local Xp = torch.Tensor({p})
    local sig11 = outer(kern, Xp, Xp)
    local sig12 = outer(kern, Xp, X)
    local sig21 = sig12:t()
    local mu    = sig12*(sig22i*Y)
    local var   = sig11 - sig12*(sig22i*sig21)

    return mu[1], var[{1, 1}]
  end
end


-- local X = torch.Tensor({0.1, 0.5, 0.9})
-- local Y = torch.Tensor({-1, 1, -1})

function krig_plot(X, Y, obj, gamma)
  local function kern(a, b)
    return sqexp(10e-4, a, b)
  end

  local f = krig(kern, X, Y, 0.1)

  local x  = torch.linspace(0, 1)
  local y  = torch.Tensor(x:size())
  local g  = torch.Tensor(x:size())
  local s  = torch.Tensor(x:size())
  local o  = torch.Tensor(x:size())
  local ym = torch.Tensor(x:size())
  local yp = torch.Tensor(x:size())

  for i=1,x:size(1) do
    local delta = 10e-9
    local alpha = torch.log(2/delta)

    local mu, var = f(x[i])
    local std = torch.sqrt(var)
    o[i]  = obj(x[i])
    y[i]  = mu
    s[i]  = std
    g[i]  = gain(mu, std, Y:max())
    -- g[i]  = std
    ym[i] = mu - std
    yp[i] = mu + std
  end

  local max_val, max_pos = g:max(1)
  local max_idx = max_pos[1]
  local arg_max = x[max_idx]
  local arg_max_std = s[max_idx]

  g = (g-g:mean())/g:std()/5
  xmax = torch.Tensor({{x[max_idx], g[max_idx]}})

  gnuplot.plot(
    {x, ym, 'lines  lt 0'},
    {x, yp, 'lines  lt 0'},
    {x, y,  'lines  lt 1'},
    {x, g,  'lines  lt 2'},
    {X, Y,  'points lt 3'},
    {x, o,  'lines  lt 4'},
    {xmax,  'points lt 8 ps 3'})

  return arg_max, y[max_idx], s[max_idx]
end

-- X = torch.linspace(0.1, 0.9, 5)
-- Y = torch.randn(X:size())
-- gamma = 0

function obj(x)
  return torch.sin(10*math.pi*x)*torch.exp(x/10)
  --return -(x - 0.5)^2
end

X = torch.Tensor({0.5})
Y = torch.Tensor({obj(0.5)})
gamma = 0

function krig_cycle()
  local x, mu, std = krig_plot(X, Y, obj, gamma)

  gamma = gamma + std^2
  local y = obj(x) + torch.randn(1)[1]*0.1
  X = X:cat(torch.Tensor({x}))
  Y = Y:cat(torch.Tensor({y}))
end
