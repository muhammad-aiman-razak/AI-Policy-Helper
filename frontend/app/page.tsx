import Chat from "../components/Chat";
import AdminPanel from "../components/AdminPanel";

export default function Page() {
  return (
    <main>
      <header style={{ marginBottom: 16 }}>
        <h1>AI Policy & Product Helper</h1>
        <p style={{ color: "#555" }}>
          Local-first RAG starter. Ingest sample docs, ask questions, and see
          citations.
        </p>
      </header>
      <AdminPanel />
      <Chat />
      <section className="card" aria-label="Instructions">
        <h3>How to test</h3>
        <ol>
          <li>
            Click <b>Ingest sample docs</b>.
          </li>
          <li>
            Ask: <i>Can a customer return a damaged blender after 20 days?</i>
          </li>
          <li>
            Ask:{" "}
            <i>
              What&apos;s the shipping SLA to East Malaysia for bulky items?
            </i>
          </li>
        </ol>
      </section>
    </main>
  );
}
