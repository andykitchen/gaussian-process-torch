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

function krig(kern, X, Y)
  assert(X:dim() == 1)
  assert(Y:dim() == 1)

  assert(X:isSameSizeAs(Y))

  local sig22  = outer(kern, X, X)
  local sig22i = torch.inverse(sig22)

  -- local u,s,v  = torch.svd(sig22)
  -- for i=1,s:size(1) do
  --   if s[i] > 1e-9 then
  --     s[i] = 1/s[i]
  --   else
  --     s[i] = 0
  --   end
  -- end

  -- local sig22i = v*torch.diag(s)*u:t()

  return function(p)
    local Xp = torch.Tensor({p})
    local sig11 = outer(kern, Xp, Xp)
    local sig12 = outer(kern, Xp, X)
    local sig21 = sig12:t()
    local mu  = sig12*(sig22i*Y)
    local var = sig11 - sig12*(sig22i*sig21)

    return mu[1], var[{1, 1}]
  end
end

-- x = torch.linspace(0, 5)
-- y = x:clone():apply(function(x) return sqexp(1, 0, x) end)
-- gnuplot.plot(x, y)

local X = torch.Tensor({1, 2.5, 4})
local Y = torch.Tensor({-1, 1, -1})

local function kern(a, b)
  return sqexp(1.0, a, b)
end

local f = krig(kern, X, Y)

x  = torch.linspace(0, 5)
y  = torch.Tensor(x:size(1)):zero()
ym = torch.Tensor(x:size(1)):zero()
yp = torch.Tensor(x:size(1)):zero()
for i=1,x:size(1) do
  local mu, var = f(x[i])
  local std = torch.sqrt(var)
  y[i]  = mu
  ym[i] = mu - std
  yp[i] = mu + std
end

gnuplot.plot(
  {x, ym, 'lines lt 0'},
  {x, yp, 'lines lt 0'},
  {x, y,  'lines lt 1'},
  {X, Y, 'points lt 3'})

-- gnuplot.plot(x, y)
-- gnuplot.imagesc(xx)
-- gnuplot.plot(xx)