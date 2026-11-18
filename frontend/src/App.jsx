
import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  LayoutDashboard,
  Map,
  Route,
  Settings,
  Bell,
  Search,
  Wifi,
  Clock3,
  Navigation,
  ChevronRight,
  Menu,
  Zap,
} from 'lucide-react'
import RouteMap3D from './components/RouteMap3D'
import { checkBackendHealth, fetchTrips, requestPrediction } from './services/api'

const EMPTY_LIST = []

function App() {
  const [trips, setTrips] = useState([])
  const [selectedTrip, setSelectedTrip] = useState('')
  const [activePage, setActivePage] = useState('Overview')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [predictionStatus, setPredictionStatus] = useState('idle')
  const [backendStatus, setBackendStatus] = useState('unknown')
  const [prediction, setPrediction] = useState(null)
  const [predictionError, setPredictionError] = useState('')
  const [predictionsRun, setPredictionsRun] = useState(0)

  useEffect(() => {
    let cancelled = false
    async function loadEvaluatorTrips() {
      try {
        await checkBackendHealth()
        const result = await fetchTrips()
        if (cancelled) return
        const availableTrips = Array.isArray(result.trip_ids) ? result.trip_ids : []
        setTrips(availableTrips)
        setSelectedTrip((current) => current && availableTrips.includes(current)
          ? current
          : (availableTrips[0] ?? ''))
        setBackendStatus('online')
        setPredictionError(availableTrips.length ? '' : 'The API has no available trips in its configured partition.')
      } catch (error) {
        if (cancelled) return
        setBackendStatus('offline')
        setPredictionError(error.response?.data?.detail ?? 'Could not connect to the PathSnap API.')
      }
    }
    loadEvaluatorTrips()
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
  const formatMetric = (value, multiplier = 1) => (
    value === null || value === undefined || !Number.isFinite(value)
      ? '--'
      : (value * multiplier).toFixed(2)
  )

  return (
    <div className="min-h-screen bg-[#080d18] text-slate-100 flex">

      {/* SIDEBAR */}
      <aside className={`${
        sidebarOpen ? 'w-64' : 'w-20'
      } shrink-0 bg-[#0c1423] border-r border-slate-800/80
      transition-all duration-300 hidden md:flex flex-col`}>

        <div className="h-20 flex items-center gap-3 px-6 border-b border-slate-800/80">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-600/20">
            <Navigation size={22} />
          </div>

          {sidebarOpen && (
            <div>
              <h1 className="text-xl font-bold tracking-wide">PathSnap</h1>
              <p className="text-[10px] text-slate-500 tracking-[2px]">
                ROUTE ANALYTICS
              </p>
            </div>
          )}
        </div>

        <div className="px-4 py-8">
          {sidebarOpen && (
            <p className="text-[10px] text-slate-500 font-semibold tracking-[2px] mb-4 px-3">
              WORKSPACE
            </p>
          )}

          {[
            { name: 'Overview', icon: LayoutDashboard },
            { name: 'Route Evaluation', icon: Route },
            { name: 'Map Explorer', icon: Map },
          ].map((item) => {
            const Icon = item.icon
            const active = activePage === item.name

            return (
              <button
                key={item.name}
                onClick={() => setActivePage(item.name)}
                title={item.name}
                className={`w-full flex items-center gap-3 px-3 py-3 mb-2 rounded-xl text-sm transition ${
                  active
                    ? 'bg-blue-600/15 text-blue-400 border border-blue-500/20'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <Icon size={19} />
                {sidebarOpen && <span>{item.name}</span>}
              </button>
            )
          })}
        </div>

        <div className="mt-auto p-4 border-t border-slate-800/80">
          <button
            onClick={() => setActivePage('Settings')}
            className="w-full flex items-center gap-3 px-3 py-3 rounded-xl text-sm text-slate-400 hover:bg-slate-800"
          >
            <Settings size={19} />
            {sidebarOpen && 'Settings'}
          </button>

          {sidebarOpen && (
            <div className="mt-5 p-4 bg-blue-500/10 border border-blue-500/10 rounded-xl">
              <Activity size={20} className="text-blue-400 mb-2" />
              <p className="text-xs font-semibold">PathSnap Evaluator</p>
              <p className="text-[11px] text-slate-500 mt-1">
                GPS Map Matching Platform
              </p>
            </div>
          )}
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 min-w-0">

        {/* TOP NAVBAR */}
        <header className="h-20 border-b border-slate-800/80 flex items-center justify-between px-5 md:px-8 bg-[#0c1423]/70">

          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 text-slate-400 hover:text-white"
            >
              <Menu size={21} />
            </button>

            <div>
              <p className="text-sm text-slate-400">
                PathSnap / {activePage}
              </p>
              <p className="text-xs text-slate-600">
                Machine Learning Evaluation
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            
<div className="hidden sm:flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800/60 border border-slate-700/50">

  <span
    className={`w-2 h-2 rounded-full ${
      backendStatus === 'online'
        ? 'bg-green-400'
        : backendStatus === 'offline'
        ? 'bg-red-400'
        : 'bg-yellow-400'
    }`}
  />

  <span className="text-xs text-slate-400">
    {backendStatus === 'online'
      ? 'Backend Online'
      : backendStatus === 'offline'
      ? 'Backend Offline'
      : 'Connection Not Checked'}
  </span>

</div>


            <button className="p-2 rounded-lg text-slate-400 hover:bg-slate-800">
              <Bell size={19} />
            </button>

            <div className="w-9 h-9 rounded-full bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 font-semibold text-sm">
              EV
            </div>
          </div>
        </header>

        {/* DASHBOARD CONTENT */}
        <div className="p-5 md:p-8 max-w-[1500px] mx-auto">

          {/* PAGE HEADING */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
            <div>
              <p className="text-blue-400 text-xs font-semibold tracking-[2px] mb-2">
                EVALUATION WORKSPACE
              </p>
              <h2 className="text-3xl font-bold tracking-tight">
                Dashboard Overview
              </h2>
              <p className="text-slate-500 text-sm mt-2">
                Monitor GPS map matching and route reconstruction.
              </p>
            </div>

            <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-800/50 border border-slate-700/50 px-4 py-3 rounded-xl">
              <Clock3 size={15} />
              Evaluation Environment
            </div>
          </div>

          {/* STAT CARDS */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-8">

            {[
              {
                label: 'Evaluation Trips',
                value: trips.length,
                detail: 'Reserved trips',
                icon: Route,
                color: 'text-blue-400',
                bg: 'bg-blue-500/10',
              },
              {
                label: 'Selected Trip',
                value: selectedTrip || '--',
                detail: 'Current selection',
                icon: Navigation,
                color: 'text-cyan-400',
                bg: 'bg-cyan-500/10',
              },
              {
                label: 'API Status',
                value: backendStatus === 'online' ? 'Online' : backendStatus === 'offline' ? 'Offline' : 'Checking',
                detail: backendStatus === 'online' ? 'Connected to FastAPI' : 'PathSnap API connection',
                icon: Wifi,
                color: backendStatus === 'online' ? 'text-green-400' : 'text-yellow-400',
                bg: backendStatus === 'online' ? 'bg-green-500/10' : 'bg-yellow-500/10',
              },
              {
                label: 'Predictions',
                value: predictionsRun,
                detail: 'Successful requests this session',
                icon: Zap,
                color: 'text-purple-400',
                bg: 'bg-purple-500/10',
              },
            ].map((stat) => {
              const Icon = stat.icon

              return (
                <div
                  key={stat.label}
                  className="bg-[#111c2e] border border-slate-800/80 rounded-2xl p-5 hover:border-slate-700 transition"
                >
                  <div className="flex justify-between items-center">
                    <p className="text-sm text-slate-400">{stat.label}</p>
                    <div className={`p-2 rounded-lg ${stat.bg} ${stat.color}`}>
                      <Icon size={19} />
                    </div>
                  </div>

                  <h3 className="text-3xl font-bold mt-5">{stat.value}</h3>
                  <p className="text-xs text-slate-500 mt-2">{stat.detail}</p>
                </div>
              )
            })}
          </div>

          {/* TRIP SELECTION */}
          <div className="bg-[#111c2e] border border-slate-800/80 rounded-2xl p-6">

            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
              <div>
                <h3 className="text-lg font-semibold">Evaluation Trips</h3>
                <p className="text-sm text-slate-500 mt-1">
                  Select a reserved trip to begin route evaluation.
                </p>
              </div>

              <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-800/60 px-3 py-2 rounded-lg">
                <Search size={14} />
                {trips.length} available trips
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">

              {trips.map((tripId, index) => {
                const number = String(index + 1).padStart(2, '0')
                const active = selectedTrip === tripId

                return (
                  <button
                    key={tripId}
                    onClick={() => {
                      setSelectedTrip(tripId)
                      setPrediction(null)
                      setPredictionStatus('idle')
                      setPredictionError('')
                    }}
                    className={`text-left p-4 rounded-xl border transition-all ${
                      active
                        ? 'bg-blue-600/10 border-blue-500/60'
                        : 'bg-[#0c1423] border-slate-800 hover:border-slate-600'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-4">
                      <span className={`text-xs font-semibold ${
                        active ? 'text-blue-400' : 'text-slate-500'
                      }`}>
                        TRIP {number}
                      </span>

                      {active && (
                        <span className="w-2 h-2 rounded-full bg-blue-400" />
                      )}
                    </div>

                    <p className="font-semibold text-sm">{tripId}</p>

                    <div className="flex items-center justify-between mt-3">
                      <span className="text-[11px] text-slate-500">
                        {prediction?.trip_id === tripId ? 'Predicted' : 'Ready'}
                      </span>
                      <ChevronRight size={15} className="text-slate-500" />
                    </div>
                  </button>
                )
              })}

              {trips.length === 0 && (
                <p className="col-span-full text-sm text-slate-500">
                  {backendStatus === 'offline' ? 'Start the API and check its data path.' : 'Loading available trips…'}
                </p>
              )}

            </div>

            <div className="mt-6 p-4 rounded-xl bg-blue-500/5 border border-blue-500/10 flex items-start gap-3">
              <Activity size={19} className="text-blue-400 mt-0.5" />
              <div>
                <p className="text-sm font-medium">Trip selection ready</p>
                <p className="text-xs text-slate-500 mt-1">
                  {selectedTrip ? `Selected trip ${selectedTrip}. The API keeps route references on the server.` : predictionError || 'Waiting for trips from the API.'}
                </p>
              </div>
            </div>

          </div>

{/* ROUTE MAP PREVIEW */}
<div className="mt-8 bg-[#111c2e] border border-slate-800/80 rounded-2xl p-6">

  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">

    <div>
      <h3 className="text-lg font-semibold">Route Visualization</h3>
      <p className="text-sm text-slate-500 mt-1">
        Geographic view of the selected evaluation trip.
      </p>
    </div>

    <div className="flex items-center gap-4 text-xs text-slate-400">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-blue-400"></span>
        GPS Points
      </div>

      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-green-400"></span>
        Predicted Route
      </div>
    </div>

  </div>

  <div className="h-[600px] rounded-xl overflow-hidden border border-slate-700/60">
     <RouteMap3D gpsPoints={gpsPoints} predictedRoute={predictedRoute} />
  </div>

  </div>



{/* EVALUATION METRICS */}
<div className="mt-8 bg-[#111c2e] border border-slate-800/80 rounded-2xl p-6">

  <div className="mb-6">
    <h3 className="text-lg font-semibold">Evaluation Metrics</h3>
    <p className="text-sm text-slate-500 mt-1">
      Performance analysis of the reconstructed route.
    </p>
  </div>

  <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">

    {[
      { name: 'Top-1 Accuracy', value: formatMetric(metrics?.viterbi_top1_accuracy, 100), unit: '%', description: 'Viterbi observation accuracy' },
      { name: 'Macro F1-score', value: formatMetric(metrics?.viterbi_macro_f1), unit: '', description: 'Viterbi macro F1' },
      { name: 'Route Continuity', value: formatMetric(metrics?.viterbi_route_continuity, 100), unit: '%', description: 'Connected road transitions' },
      { name: 'Inference Time', value: formatMetric(prediction?.inference_time_seconds), unit: 's', description: 'Prediction processing time' },
    ].map((metric) => (
      <div
        key={metric.name}
        className="bg-[#0c1423] border border-slate-800 rounded-xl p-5"
      >
        <p className="text-sm text-slate-400">{metric.name}</p>

        <h4 className="text-3xl font-bold mt-5 text-slate-200">
          {metric.value}
          <span className="text-sm text-slate-500 ml-1">
            {metric.unit}
          </span>
        </h4>

        <p className="text-xs text-slate-500 mt-2">
          {metric.description}
        </p>
      </div>
    ))}

  </div>

  <div className="mt-5 p-4 rounded-xl bg-slate-800/40 border border-slate-700/50">
    <p className="text-xs text-slate-400">
      {metrics ? `${metrics.evaluated_observations} labeled observations evaluated · Candidate miss rate ${formatMetric(metrics.candidate_miss_rate, 100)}%. Reference route labels remain server-side.` : prediction ? 'No server-side reference route is available for this trip; prediction is returned without evaluation metrics.' : 'Run a prediction to display its evaluation metrics.'}
    </p>
  </div>

</div>

{/* PREDICTION CONTROL PANEL */}
<div className="mt-8 bg-[#111c2e] border border-slate-800/80 rounded-2xl p-6">

  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-5">

    <div>
      <p className="text-blue-400 text-xs font-semibold tracking-widest mb-2">
        ROUTE RECONSTRUCTION
      </p>

      <h3 className="text-lg font-semibold">
        Run Route Prediction
      </h3>

      <p className="text-sm text-slate-500 mt-2">
        Process the selected GPS trajectory and reconstruct its most likely route.
      </p>
    </div>

    <button
      onClick={runPrediction}
      disabled={!selectedTrip || backendStatus !== 'online' || predictionStatus === 'loading'}
      className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-semibold transition shadow-lg shadow-blue-600/20"
    >
      <Zap size={18} />
      {predictionStatus === 'loading' ? 'Predicting…' : 'Run Prediction'}
    </button>

  </div>

  <div className="mt-6 p-4 rounded-xl bg-[#0c1423] border border-slate-800">

    <div className="flex items-center gap-3">

      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${predictionStatus === 'error' ? 'bg-red-500/10' : predictionStatus === 'success' ? 'bg-green-500/10' : 'bg-yellow-500/10'}`}>
        <Activity size={20} className={predictionStatus === 'error' ? 'text-red-400' : predictionStatus === 'success' ? 'text-green-400' : 'text-yellow-400'} />
      </div>

      <div>
        <p className="text-sm font-medium">
          {predictionStatus === 'loading' ? 'Prediction running'
            : predictionStatus === 'success' ? 'Prediction complete'
              : predictionStatus === 'error' ? 'Prediction failed'
                : backendStatus === 'online' ? 'Ready for evaluation' : 'Connecting to backend'}
        </p>

        <p className="text-xs text-slate-500 mt-1">
          {predictionError || (prediction
            ? `${prediction.observation_count} GPS observations · ${prediction.predicted_route.length} predicted route segments`
            : selectedTrip ? `Selected trip ${selectedTrip} · ${backendStatus === 'online' ? 'Ready to predict' : 'Waiting for API connection'}` : 'Waiting for an available trip from the API.')}
        </p>
      </div>

    </div>

  </div>

</div>

{/* PREDICTION RESULTS */}
<div className="mt-8 bg-[#111c2e] border border-slate-800/80 rounded-2xl p-6">

  <div className="mb-6">
    <p className="text-blue-400 text-xs font-semibold tracking-widest mb-2">
      INFERENCE OUTPUT
    </p>

    <h3 className="text-lg font-semibold">
      Prediction Results
    </h3>

    <p className="text-sm text-slate-500 mt-1">
      Detailed output from the route reconstruction pipeline.
    </p>
  </div>

  {/* RESULT SUMMARY */}
  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">

    {[
      {
        label: 'Selected Trip',
        value: selectedTrip || '--',
      },
      {
        label: 'GPS Observations',
        value: prediction?.observation_count ?? '--',
      },
      {
        label: 'Predicted Segments',
        value: prediction?.predicted_route?.length ?? '--',
      },
    ].map((item) => (
      <div
        key={item.label}
        className="bg-[#0c1423] border border-slate-800 rounded-xl p-5"
      >
        <p className="text-xs text-slate-500">
          {item.label}
        </p>

        <h4 className="text-xl font-semibold mt-3">
          {item.value}
        </h4>
      </div>
    ))}

  </div>

  {/* PREDICTED ROAD SEQUENCE */}
  <div className="bg-[#0c1423] border border-slate-800 rounded-xl p-5">

    <div className="flex items-center justify-between mb-4">
      <h4 className="font-semibold text-sm">
        Predicted Road Sequence
      </h4>

      <span className="text-xs text-slate-500">
        HMM + Viterbi Output
      </span>
    </div>

    {prediction ? (
      <div>
        <div className="flex flex-wrap gap-2">
          {prediction.predicted_route.slice(0, 30).map((segment, index) => (
            <span key={`${segment.arc_id}-${index}`} className="px-2.5 py-1.5 rounded-lg bg-green-500/10 border border-green-500/20 text-green-300 text-xs">
              {segment.arc_id}
            </span>
          ))}
          {prediction.predicted_route.length > 30 && (
            <span className="px-2.5 py-1.5 text-slate-500 text-xs">
              +{prediction.predicted_route.length - 30} more
            </span>
          )}
        </div>
        <p className="text-xs text-slate-500 mt-4">
          Classifier-only and Viterbi sequences are available in the API response. Mean emission confidence: {formatMetric(prediction.mean_emission_confidence)}.
        </p>
      </div>
    ) : (
      <div className="min-h-24 flex flex-col items-center justify-center text-center">
        <div className="w-12 h-12 rounded-xl bg-slate-800 flex items-center justify-center mb-3">
          <Route size={22} className="text-slate-400" />
        </div>
        <p className="text-sm text-slate-400">No prediction available</p>
        <p className="text-xs text-slate-600 mt-1">
          Select a trip and run the prediction to see its road sequence.
        </p>
      </div>
    )}

  </div>

  {/* PROCESSING INFORMATION */}
  <div className="mt-4 flex items-center gap-2 text-xs text-slate-500">
    <Activity size={15} />
    {prediction ? `Classifier-only and Viterbi outputs received for ${prediction.trip_id}.` : 'Prediction output will appear here after successful inference.'}
  </div>

</div>

          {/* FOOTER */}
          <footer className="mt-8 text-center text-xs text-slate-600">
            PathSnap · Distributed Map-Matching & Route Reconstruction
          </footer>

        </div>
      </main>
    </div>
  )
}

export default App
