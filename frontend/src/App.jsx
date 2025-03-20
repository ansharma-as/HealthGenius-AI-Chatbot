import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./App.css";
import HomePage from "./page/Home";
import Contact from "./page/Contact";
import ChatPage from "./page/ChatPage";
import AboutUs from "./page/AboutUs";

function App() {
  return (
    <div className="App">
      {/* Enable both flags */}
      <Router
        future={{
          v7_startTransition: true,  // Opt-in to concurrent rendering
          v7_relativeSplatPath: true, // Opt-in to new relative splat path resolution
        }}
      >
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/about" element={<AboutUs />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
