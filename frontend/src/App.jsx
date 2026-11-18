import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  ArrowUpRight,
  Clock3,
  Crosshair,
  MapPin,
  Navigation,
  Route as RouteIcon,
  Search,
  Wifi,
  Zap,
} from 'lucide-react'
import RouteMap from './components/RouteMap'
import { checkBackendHealth, fetchTrips, requestPrediction } from './services/api'

const EMPTY_LIST = []
const panel = 'rounded-2xl border border-white/[0.08] bg-[#0b0d0c]'

function App() {
  const [trips, setTrips] = useState([])
  const [selectedTrip, setSelectedTrip] = useState('')
  const [predictionStatus, setPredictionStatus] = useState('idle')
  const [backendStatus, setBackendStatus] = useState('unknown')
  const [prediction, setPrediction] = useState(null)
  const [predictionError, setPredictionError] = useState('')
  const [predictionsRun, setPredictionsRun] = useState(0)

  useEffect(() => {
    let cancelled = false
    async function loadTrips() {
      try {
        await checkBackendHealth()
        const result = await fetchTrips()
        if (cancelled) return
        const available = Array.isArray(result.trip_ids) ? result.trip_ids : []
        setTrips(available)
        setSelectedTrip((current) => current && available.includes(current) ? current : (available[0] ?? ''))
        setBackendStatus('online')
        setPredictionError(available.length ? '' : 'The API has no trips in its configured partition.')
      } catch (error) {
        if (cancelled) return
        setBackendStatus('offline')
        setPredictionError(error.response?.data?.detail ?? 'Could not connect to the PathSnap API.')
      }
    }
    loadTrips()
    return () => { cancelled = true }
  }, [])

  async function runPrediction() {
    if (!selectedTrip || predictionStatus === 'loading') return
    setPredictionStatus('loading')
    setPredictionError('')
    setPrediction(null)
    try {
      const result = await requestPrediction(selectedTrip)
      setPrediction(result)
      setPredictionStatus('success')
      setPredictionsRun((count) => count + 1)
    } catch (error) {
      setPredictionStatus('error')
      setPredictionError(error.response?.data?.detail ?? error.message ?? 'Prediction failed.')
    }
  }

  const gpsPoints = useMemo(
    () => prediction?.observations?.map((observation) => observation.gps) ?? EMPTY_LIST,
    [prediction],
  )
  const predictedRoute = prediction?.predicted_route ?? EMPTY_LIST
  const metrics = prediction?.evaluation
  const metric = (value, multiplier = 1) => value == null || !Number.isFinite(value)
    ? '--'
    : (value * multiplier).toFixed(2)

  const stats = [
    { label: 'Available trips', value: trips.length, foot: 'In selected partition', icon: RouteIcon },
    { label: 'Predictions run', value: predictionsRun, foot: 'This session', icon: Zap },
    { label: 'Selected trip', value: selectedTrip || '—', foot: 'Ready for evaluation', icon: Navigation },
  ]

  return (
    <div className="min-h-screen bg-[#050606] text-zinc-100">
      <header className="sticky top-0 z-20 border-b border-white/[0.08] bg-[#050606]/90 backdrop-blur-xl">
        <div className="mx-auto flex h-[76px] max-w-[1600px] items-center justify-between px-5 md:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#b6ff3b] text-black shadow-[0_0_28px_rgba(182,255,59,.2)]">
              <Crosshair size={21} />
            </div>
            <div>
              <p className="text-base font-bold tracking-tight">PATH<span className="text-[#b6ff3b]">SNAP</span></p>
              <p className="text-[10px] font-medium tracking-[.22em] text-zinc-500">ROUTE INTELLIGENCE</p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">
            <span className={`h-2 w-2 rounded-full ${backendStatus === 'online' ? 'bg-[#b6ff3b] shadow-[0_0_10px_#b6ff3b]' : backendStatus === 'offline' ? 'bg-red-400' : 'bg-amber-300'}`} />
            <span className="text-xs text-zinc-300">{backendStatus === 'online' ? 'API connected' : backendStatus === 'offline' ? 'API offline' : 'Connecting'}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] px-5 py-8 md:px-8 md:py-10">
        <section className="mb-8 flex flex-col justify-between gap-5 md:flex-row md:items-end">
          <div>
            <p className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[.24em] text-[#b6ff3b]"><Activity size={14} /> Evaluation workspace</p>
            <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Map matching <span className="text-zinc-500">dashboard</span></h1>
            <p className="mt-2 max-w-2xl text-sm text-zinc-500">Inspect trips, run route reconstruction, and review model performance.</p>
          </div>
          <div className="flex items-center gap-2 text-xs text-zinc-400"><Clock3 size={15} className="text-[#b6ff3b]" /> Live evaluation environment</div>
        </section>

        <section className="mb-6 grid gap-3 sm:grid-cols-3">
          {stats.map(({ label, value, foot, icon: Icon }) => (
            <div key={label} className={`${panel} flex items-center justify-between p-5`}>
              <div><p className="text-xs text-zinc-500">{label}</p><p className="mt-2 text-2xl font-semibold tracking-tight">{value}</p><p className="mt-1 text-[11px] text-zinc-600">{foot}</p></div>
              <div className="rounded-xl border border-[#b6ff3b]/15 bg-[#b6ff3b]/[0.07] p-3 text-[#b6ff3b]"><Icon size={19} /></div>
            </div>
          ))}
        </section>

        <section className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
          <div className={`${panel} p-5`}>
            <div className="mb-5 flex items-center justify-between">
              <div><h2 className="font-semibold">Trip explorer</h2><p className="mt-1 text-xs text-zinc-500">Choose a trip to inspect</p></div>
              <span className="rounded-full bg-white/[0.05] px-2.5 py-1 text-[11px] text-zinc-400">{trips.length} trips</span>
            </div>
            <div className="mb-3 flex items-center gap-2 rounded-lg border border-white/[0.08] bg-black/40 px-3 py-2.5 text-xs text-zinc-600"><Search size={14} /> Trip IDs from API</div>
            <div className="max-h-[440px] space-y-2 overflow-auto pr-1">
              {trips.map((tripId, index) => {
                const active = selectedTrip === tripId
                return <button key={tripId} onClick={() => { setSelectedTrip(tripId); setPrediction(null); setPredictionStatus('idle'); setPredictionError('') }} className={`group flex w-full items-center justify-between rounded-xl border px-3.5 py-3 text-left transition ${active ? 'border-[#b6ff3b]/40 bg-[#b6ff3b]/[0.08]' : 'border-white/[0.06] bg-white/[0.02] hover:border-white/15'}`}>
                  <span className="flex items-center gap-3"><span className={`flex h-8 w-8 items-center justify-center rounded-lg text-[10px] font-semibold ${active ? 'bg-[#b6ff3b] text-black' : 'bg-white/[0.06] text-zinc-500'}`}>{String(index + 1).padStart(2, '0')}</span><span><span className={`block text-sm font-medium ${active ? 'text-[#d8ff9b]' : 'text-zinc-300'}`}>{tripId}</span><span className="mt-0.5 block text-[10px] text-zinc-600">{prediction?.trip_id === tripId ? 'Prediction ready' : 'Trip data available'}</span></span></span>
                  <ArrowUpRight size={15} className={active ? 'text-[#b6ff3b]' : 'text-zinc-700 group-hover:text-zinc-400'} />
                </button>
              })}
              {!trips.length && <p className="rounded-xl border border-dashed border-white/10 p-4 text-xs leading-5 text-zinc-500">{predictionError || 'Waiting for trips from the API.'}</p>}
            </div>
            <button onClick={runPrediction} disabled={!selectedTrip || backendStatus !== 'online' || predictionStatus === 'loading'} className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-[#b6ff3b] px-4 py-3 text-sm font-bold text-black transition hover:bg-[#c9ff70] disabled:cursor-not-allowed disabled:opacity-40"><Zap size={16} />{predictionStatus === 'loading' ? 'Reconstructing route…' : 'Run route prediction'}</button>
            {(predictionError || predictionStatus === 'success') && <p className={`mt-3 text-xs leading-5 ${predictionStatus === 'error' ? 'text-red-300' : 'text-[#b6ff3b]'}`}>{predictionError || `Prediction complete for ${prediction?.trip_id}.`}</p>}
          </div>

          <div className={`${panel} min-w-0 p-5`}>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div><div className="flex items-center gap-2"><MapPin size={16} className="text-[#b6ff3b]" /><h2 className="font-semibold">Route visualization</h2></div><p className="mt-1 text-xs text-zinc-500">{prediction ? `Trip ${prediction.trip_id} · ${prediction.observation_count} GPS observations` : 'Select a trip and run a prediction to draw its route.'}</p></div>
              <div className="flex gap-4 text-[11px] text-zinc-400"><span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-blue-400" />GPS points</span><span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-full bg-[#b6ff3b]" />Predicted route</span></div>
            </div>
            <div className="h-[520px] overflow-hidden rounded-xl border border-white/10 bg-black"><RouteMap gpsPoints={gpsPoints} predictedRoute={predictedRoute} /></div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1.2fr_.8fr]">
          <div className={`${panel} p-5`}>
            <div className="mb-5 flex items-start justify-between"><div><h2 className="font-semibold">Evaluation metrics</h2><p className="mt-1 text-xs text-zinc-500">Server-side reference comparison, when available</p></div><span className="rounded-full border border-[#b6ff3b]/20 px-2.5 py-1 text-[10px] uppercase tracking-wider text-[#b6ff3b]">{metrics ? 'Evaluated' : 'Awaiting run'}</span></div>
            <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
              {[
                ['Top-1 accuracy', metric(metrics?.viterbi_top1_accuracy, 100), '%'],
                ['Macro F1', metric(metrics?.viterbi_macro_f1), ''],
                ['Route continuity', metric(metrics?.viterbi_route_continuity, 100), '%'],
                ['Inference time', metric(prediction?.inference_time_seconds), 's'],
              ].map(([name, value, unit]) => <div key={name} className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4"><p className="text-[11px] text-zinc-500">{name}</p><p className="mt-3 text-2xl font-semibold">{value}<span className="ml-1 text-xs font-normal text-zinc-500">{unit}</span></p></div>)}
            </div>
            <p className="mt-4 text-[11px] leading-5 text-zinc-600">{metrics ? `${metrics.evaluated_observations} labeled observations · Candidate miss rate ${metric(metrics.candidate_miss_rate, 100)}%. Reference labels stay on the API server.` : prediction ? 'No server-side reference route is available for this trip.' : 'Run a prediction to populate metrics.'}</p>
          </div>
          <div className={`${panel} p-5`}>
            <div className="mb-4 flex items-center justify-between"><div><h2 className="font-semibold">Predicted road sequence</h2><p className="mt-1 text-xs text-zinc-500">HMM + Viterbi output</p></div><span className="text-xs text-zinc-600">{predictedRoute.length} segments</span></div>
            {predictedRoute.length ? <div className="flex max-h-40 flex-wrap gap-2 overflow-auto">{predictedRoute.slice(0, 36).map((segment, index) => <span key={`${segment.arc_id}-${index}`} className="rounded-lg border border-[#b6ff3b]/20 bg-[#b6ff3b]/[0.06] px-2.5 py-1.5 font-mono text-[11px] text-[#caff83]">{segment.arc_id}</span>)}</div> : <div className="flex min-h-24 items-center justify-center rounded-xl border border-dashed border-white/[0.08] text-xs text-zinc-600">No prediction yet</div>}
            {prediction && <p className="mt-4 text-[11px] text-zinc-600">Mean emission confidence: {metric(prediction.mean_emission_confidence)}</p>}
          </div>
        </section>

        <footer className="flex flex-col gap-2 py-8 text-center text-[10px] uppercase tracking-[.18em] text-zinc-700 sm:flex-row sm:justify-between"><span>PathSnap · Route intelligence</span><span className="flex items-center justify-center gap-1.5"><Wifi size={12} /> {backendStatus === 'online' ? 'API online' : backendStatus === 'offline' ? 'API offline' : 'Checking API'}</span></footer>
      </main>
    </div>
  )
}

export default App
