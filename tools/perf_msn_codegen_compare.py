"""Quick performance comparison between Brian2 codegen targets: 'numpy' vs 'cython'.

This script runs the MSN simulation twice for each backend and reports timing.
"""
import time
from brian2 import *

# MSN parameters (copied minimal set)
C_Cm     = 10e-7
C_Ra     = 47
C_Rm_hi  = 100e3
C_Rm_lo  = 500
C_Vth    = 1.5
C_Ihold  = 100e-6
C_I_drive = 92.4e-6


def run_msn(duration_ms=300, codegen_target='numpy'):
    prefs.codegen.target = codegen_target
    start_scope()
    defaultclock.dt = 1*us
    Cm     = C_Cm    * farad
    Ra     = C_Ra    * ohm
    Rm_hi  = C_Rm_hi * ohm
    Rm_lo  = C_Rm_lo * ohm
    Vth_q  = C_Vth   * volt
    Ihold_q = C_Ihold * amp
    Iin_q  = C_I_drive * amp
    eqs = '''
    dVm/dt = (Iin_q - Vm/(Rm_S + Ra)) / Cm   : volt
    Rm_S   = (1 - s)*Rm_hi + s*Rm_lo          : ohm
    I_M    = Vm / (Rm_S + Ra)                 : amp
    Vout   = Vm * Ra / (Rm_S + Ra)            : volt
    s      : 1
    '''
    G = NeuronGroup(1, eqs,
                    threshold='Vm > Vth_q and s < 0.5',
                    reset='s = 1',
                    events={'reopen': 'I_M < Ihold_q and s > 0.5'},
                    method='euler')
    G.run_on_event('reopen', 's = 0')
    G.Vm = 0*volt
    G.s  = 0
    sm = StateMonitor(G, ['Vm', 'Vout'], record=True, dt=2*us)
    sp = SpikeMonitor(G)
    t0 = time.time()
    run(duration_ms*ms)
    t1 = time.time()
    return t1 - t0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()

    results = {}
    for target in ('numpy', 'cython'):
        times = []
        for i in range(args.repeats):
            print(f"Running MSN ({target}) trial {i+1}...")
            try:
                elapsed = run_msn(codegen_target=target)
            except Exception as e:
                print(f"Error running with target {target}: {e}")
                elapsed = None
            times.append(elapsed)
        results[target] = times

    print('\nResults:')
    for target, times in results.items():
        print(f"  {target}: {times}")
