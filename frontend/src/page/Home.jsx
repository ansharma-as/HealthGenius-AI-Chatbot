import React from "react";
import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import InfoSection from "../components/InfoSection";
import { useNavigate } from "react-router-dom";

const HomePage = () => {
  const navigate = useNavigate();

  const handleNavigation = () => {
    navigate("/chat");
  };

  return (
    <div className=" text-black">
      <motion.div
        initial={{ opacity: 0, y: -100 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Navbar />
      </motion.div>

      {/* Hero Section */}
      <section className="flex flex-col md:flex-row items-center justify-between px-8 md:px-16 pt-28">
        <motion.div
          initial={{ opacity: 0, x: -100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className="w-full md:w-1/2 space-y-4"
        >
          <h2 className="text-gray-600 font-semibold uppercase">
            Purpose of AI Doctor
          </h2>
          <h1 className="text-4xl md:text-6xl font-bold leading-tight">
            Detailed diagnostic <br />
            of your body
          </h1>
          <p className="text-gray-600">
            Health is the most important thing. So don't put it off for later.
            Think about your future today.
          </p>
          <button
            onClick={handleNavigation}
            className="mt-4 px-6 py-2 border border-gray-800 text-gray-800 rounded hover:bg-gray-800 hover:text-green-400 transition"
          >
            Start chat
          </button>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className="w-full md:w-1/2 mt-8 md:mt-0"
        >
          <img
            src="/I1-transformed.avif"
            alt="DNA Graphic"
            className="w-full h-auto object-cover rounded-xl bg-blend-color-burn"
          />
        </motion.div>
      </section>

      {/* Info Section */}
      <InfoSection />

      <Footer />
    </div>
  );
};

export default HomePage;
