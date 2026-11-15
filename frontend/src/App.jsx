import { Route, Routes, useLocation } from "react-router-dom";
import Sidebar from "./components/Sidebar.jsx";
import useHealth from "./hooks/useHealth.js";
import Home from "./pages/Home.jsx";
import TextCheck from "./pages/TextCheck.jsx";
import ImageCheck from "./pages/ImageCheck.jsx";
import ClaimCheck from "./pages/ClaimCheck.jsx";
import BatchCheck from "./pages/BatchCheck.jsx";
import History from "./pages/History.jsx";

export default function App() {
  const { pathname } = useLocation();
  const warmup = useHealth()?.data?.warmup;
  return (
    <div className="min-h-screen lg:flex">
      <Sidebar />
      <main className="min-w-0 flex-1 px-4 py-8 sm:px-8 lg:px-12 lg:py-12">
        {warmup?.running && (
          <div className="no-print mx-auto mb-6 max-w-[1180px] rounded-lg border border-rule-soft px-4 py-3 text-sm" role="status">
            <span className="font-medium">Getting the AI models ready: {warmup.done} of {warmup.total}</span>
            {warmup.current && <span className="text-graphite"> (loading the {warmup.current.toLowerCase()})</span>}.{" "}
            <span className="text-graphite">
              The very first start also downloads them once, which can take a while. You can paste text meanwhile.
            </span>
          </div>
        )}
        <div key={pathname} className="page-in mx-auto max-w-[1180px]">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/text" element={<TextCheck />} />
            <Route path="/image" element={<ImageCheck />} />
            <Route path="/claims" element={<ClaimCheck />} />
            <Route path="/batch" element={<BatchCheck />} />
            <Route path="/history" element={<History />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
