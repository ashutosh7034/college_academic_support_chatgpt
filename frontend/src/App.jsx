import { useMemo, useState } from 'react'

const API_BASE = 'http://127.0.0.1:8000'

export default function App() {
  const [files, setFiles] = useState([])
  const [processing, setProcessing] = useState(false)
  const [asking, setAsking] = useState(false)
  const [indexReady, setIndexReady] = useState(false)
  const [indexedChunks, setIndexedChunks] = useState(0)
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(4)
  const [status, setStatus] = useState('Upload PDFs and build index to start asking questions.')
  const [result, setResult] = useState(null)

  const selectedLabel = useMemo(() => {
    if (!files.length) return 'No files selected'
    if (files.length === 1) return files[0].name
    return `${files.length} files selected`
  }, [files])

  const handleFilesChange = (event) => {
    const selected = Array.from(event.target.files ?? [])
    setFiles(selected)
  }

  const processFiles = async () => {
    if (!files.length) {
      setStatus('Please select at least one PDF file.')
      return
    }

    setProcessing(true)
    setResult(null)
    setStatus('Building index...')

    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))

    try {
      const response = await fetch(`${API_BASE}/index`, {
        method: 'POST',
        body: formData
      })
      const data = await response.json()

      if (!data.ok) {
        setIndexReady(false)
        setIndexedChunks(0)
        setStatus(data.message || 'Indexing failed.')
        return
      }

      setIndexReady(true)
      setIndexedChunks(data.indexed_chunks || 0)
      setStatus(`Index ready. ${data.indexed_chunks} chunks created.`)
    } catch (error) {
      setIndexReady(false)
      setStatus('Failed to connect to backend API.')
    } finally {
      setProcessing(false)
    }
  }

  const askQuestion = async () => {
    if (!indexReady) {
      setStatus('Index not ready. Process PDFs first.')
      return
    }

    if (!question.trim()) {
      setStatus('Please type a question.')
      return
    }

    setAsking(true)
    setStatus('Retrieving answer...')

    const formData = new FormData()
    formData.append('question', question)
    formData.append('top_k', String(topK))

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        body: formData
      })
      const data = await response.json()

      if (!data.ok) {
        setStatus(data.message || 'Question answering failed.')
        setResult(null)
        return
      }

      setResult(data.result)
      setStatus('Answer ready.')
    } catch (error) {
      setStatus('Failed to connect to backend API.')
      setResult(null)
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="page">
      <div className="glow one" />
      <div className="glow two" />

      <main className="card">
        <header className="header">
          <h1>College Academic Support Chatbot</h1>
          <p>Ask questions grounded in your uploaded documents.</p>
        </header>

        <section className="panel">
          <div className="row">
            <label className="uploadBtn">
              <input type="file" accept="application/pdf" multiple onChange={handleFilesChange} />
              Select PDF Files
            </label>
            <button className="primary" onClick={processFiles} disabled={processing}>
              {processing ? 'Processing...' : 'Process PDFs'}
            </button>
          </div>
          <div className="fileMeta">{selectedLabel}</div>
          <div className="indexMeta">
            <span className={indexReady ? 'badge success' : 'badge muted'}>
              {indexReady ? 'Index Ready' : 'Not Indexed'}
            </span>
            <span className="badge">Chunks: {indexedChunks}</span>
          </div>
        </section>

        <section className="panel">
          <label className="label">Ask Question</label>
          <textarea
            rows={3}
            placeholder="e.g., Which department issued this document?"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />

          <div className="sliderRow">
            <label>Retrieved Chunks: {topK}</label>
            <input
              type="range"
              min={1}
              max={8}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </div>

          <button className="primary" onClick={askQuestion} disabled={asking || !indexReady}>
            {asking ? 'Getting Answer...' : 'Get Answer'}
          </button>
        </section>

        <section className="status">{status}</section>

        {result && (
          <section className="panel answer">
            <h2>Answer</h2>
            <p className="answerText">{result.answer}</p>

            <div className="stats">
              <div><strong>Confidence:</strong> {Number(result.confidence).toFixed(3)}</div>
              <div>
                <strong>Best source:</strong> {result.best_source} {result.best_page ? `(Page ${result.best_page})` : ''}
              </div>
            </div>

            <h3>Sources</h3>
            <ul>
              {(result.sources || []).map((source) => (
                <li key={source}>{source}</li>
              ))}
            </ul>

            <h3>Retrieved Context</h3>
            <div className="contexts">
              {(result.contexts || []).map((ctx) => (
                <article key={ctx.chunk_id} className="contextCard">
                  <div className="contextHead">
                    <span>{ctx.source} • Page {ctx.page}</span>
                    <span>Score {Number(ctx.score).toFixed(3)}</span>
                  </div>
                  <p>{ctx.text}</p>
                </article>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}
