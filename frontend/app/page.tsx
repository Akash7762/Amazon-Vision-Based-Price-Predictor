import { PricePredictor } from "@/components/PricePredictor";

export default function Home() {
  return (
    <main className="page">
      <header className="masthead">
        <h1>Vision Price Predictor</h1>
        <p>Photograph a product to get an estimate of what it sells for.</p>
      </header>

      <PricePredictor />

      <footer className="footnote">
        Estimates come from a model trained on about 51,000 product photos and their listed prices. An academic
        project, not affiliated with Amazon.
      </footer>
    </main>
  );
}
