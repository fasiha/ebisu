import json
import pylab as plt  #type:ignore

plt.style.use('ggplot')
plt.rcParams['svg.fonttype'] = 'none'
plt.ion()

if __name__ == "__main__":
  with open('beta-powerlaw-compare.json', 'r') as fid:
    hlsPowerlaw = json.load(fid)
  with open('ensemble-compare-halflives.json', 'r') as fid:
    hlsEnsemble = json.load(fid)

  zipped = list(zip(hlsEnsemble["0.5"], hlsEnsemble["0.8"], hlsPowerlaw["0.5"], hlsPowerlaw["0.8"]))

  en50, en80, po50, po80 = list(zip(*sorted(zipped, key=lambda v: v[0])))
  en50_2, en80_2, po50_2, po80_2 = list(zip(*sorted(zipped, key=lambda v: v[2])))

  fig, axs = plt.subplots(1, 2)
  axs[0].semilogy(en50, 'C0-', label="ensemble p=0.5")
  axs[0].semilogy(en80, 'C0--', label="ensemble p=0.8")
  axs[0].semilogy(po50, 'C1-', label="powerlaw p=0.5")
  axs[0].semilogy(po80, 'C1--', label="powerlaw p=0.8")

  axs[1].semilogy(en50_2, 'C0-', label="ensemble p=0.5")
  axs[1].semilogy(en80_2, 'C0--', label="ensemble p=0.8")
  axs[1].semilogy(po50_2, 'C1-', label="powerlaw p=0.5")
  axs[1].semilogy(po80_2, 'C1--', label="powerlaw p=0.8")

  for a in axs:
    a.legend()
    a.set_xlabel('flashcard')
  axs[0].set_ylabel('Hours till decay')
  fig.suptitle('Comparing beta-powerlaw and v3-ensemble final halflife')

  fig.tight_layout()
  plt.savefig('comparing-halflives.png', dpi=300)
  plt.savefig('comparing-halflives.svg')
