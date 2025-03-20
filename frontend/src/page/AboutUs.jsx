import React from "react";
import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

const AboutUs = () => {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Navbar */}
      <header className="bg-transparent">
        <Navbar />
      </header>

      {/* About Us Section */}
      <main className="flex-grow">
        <motion.div
          className="relative bg-cover bg-center"
          style={{ backgroundImage: "url('/B1-transformed.jpeg')" }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
        >
          {/* Optional Overlay */}
          <div className="absolute inset-0 bg-black bg-opacity-50"></div>

          {/* Content */}
          <motion.div
            className="relative z-10 text-white text-center py-16 px-6 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
          >
            <h1 className="text-5xl font-bold mb-6">About Us</h1>
            <p className="text-lg leading-relaxed mb-6">
              Welcome to <span className="font-semibold">AI Doctor</span>, your trusted partner in harnessing the power of artificial intelligence for simplified and efficient healthcare solutions. Our mission is to empower individuals and healthcare providers with cutting-edge tools that make diagnostics and treatment planning faster, more accessible, and highly accurate.
            </p>
            <p className="text-lg leading-relaxed mb-6">
              By leveraging advanced machine learning models and intuitive user interfaces, AI Doctor aims to bridge the gap between technology and healthcare. Whether you're seeking recommendations for medications or guidance on medical tests, we’re here to provide insightful, AI-driven assistance every step of the way.
            </p>
            <p className="text-lg leading-relaxed">
              Together, let's reimagine the future of healthcare with innovation, empathy, and precision. AI Doctor is committed to making healthcare smarter, simpler, and more inclusive for everyone.
            </p>
          </motion.div>
        </motion.div>

        {/* Why Choose Us Section */}
        <motion.section
          className="py-16 bg-gray-50"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
        >
          <div className="max-w-5xl mx-auto text-center px-6">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">Why Choose Us?</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8 }}
              >
                <h3 className="text-xl font-semibold text-blue-600 mb-3">AI-Powered Insights</h3>
                <p className="text-gray-700">
                  Our platform uses state-of-the-art AI technology to provide you with accurate and reliable medical insights.
                </p>
              </motion.div>
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.4 }}
              >
                <h3 className="text-xl font-semibold text-blue-600 mb-3">User-Centric Design</h3>
                <p className="text-gray-700">
                  We prioritize simplicity and ease of use, ensuring a seamless experience for every user, regardless of technical expertise.
                </p>
              </motion.div>
              <motion.div
                className="p-6 bg-white rounded-lg shadow-md"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.8 }}
              >
                <h3 className="text-xl font-semibold text-blue-600 mb-3">Accessible Healthcare</h3>
                <p className="text-gray-700">
                  Our tools are designed to make healthcare accessible to all, breaking down barriers to medical information and diagnostics.
                </p>
              </motion.div>
            </div>
          </div>
        </motion.section>
      </main>

      {/* Footer */}
      <footer>
        <Footer />
      </footer>
    </div>
  );
};

export default AboutUs;
