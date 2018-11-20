function [lo, hi] = bnd(a, b, check)
  if ~exist('check', 'var') || isempty(check), check = false; end
  assert(all(a(:) < b(:)));
  delta = b - a;
  k = floor(delta);
  s = delta - k;
  if numel(a) == 1
    c = 1 ./ prod(a + s - 1 + (0 : k));
  else
    c = arrayfun(@(k, s, a) 1 ./ prod(a + s - 1 + (0 : k)), k, s, a);
  end
  lo = c .* max(a - 1, 0) .^ (1 - s);
  hi = c .* a .^ (1 - s);
  
  if check
    lexpected = gammaln(a) - gammaln(b);
    assert(log(lo) < lexpected);
    assert(lexpected < log(hi));
  end
end

