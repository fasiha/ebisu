function [d] = compareSorted(y, y2)
  [~, s] = sort(y);
  [~, t] = sort(s);
  [~, s2] = sort(y2);
  [~, t2] = sort(s2);
  d = t - t2;
end


