import { Route, Routes, useLocation } from "react-router-dom";
import Sidebar from "./components/Sidebar.jsx";
import Home from "./pages/Home.jsx";
import TextCheck from "./pages/TextCheck.jsx";
import ImageCheck from "./pages/ImageCheck.jsx";
import ClaimCheck from "./pages/ClaimCheck.jsx";
import BatchCheck from "./pages/BatchCheck.jsx";
import History from "./pages/History.jsx";

export default function App() {
  const { pathname } = useLocation();
  return (
    <div className="min-h-screen lg:flex">
      <Sidebar />
      <main className="min-w-0 flex-1 px-4 py-8 sm:px-8 lg:px-12 lg:py-12">
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
