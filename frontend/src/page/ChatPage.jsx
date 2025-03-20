import { motion } from "framer-motion";
import React, { useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Popup from "reactjs-popup";
import "reactjs-popup/dist/index.css";

const ChatPage = () => {
  const [report, setReport] = useState({});
  const [formData, setFormData] = useState({
    user_input: "",
  });
  const [isPopupOpen, setIsPopupOpen] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(import.meta.env.VITE_BACKEND_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_input: formData.user_input,
        }),
      });
      const data = await response.json();
      setReport(data);
      setIsPopupOpen(true);
      toast.success("Your Full Report is ready");
    } catch (error) {
      toast.error("Failed to fetch the report. Please try again.");
    }
  };

  const handlePopupOpen = () => {
    setIsPopupOpen(true);
  };

  return (
    <div className="relative w-full max-h-full bg-cover bg-center" style={{ backgroundImage: "url('/B1-transformed.jpeg')" }}>
      {/* Optional Overlay */}
      <div className="absolute inset-0 bg-black bg-opacity-50"></div>

      {/* Navbar */}
      <motion.nav
        initial={{ opacity: 0, y: -100 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-50"
      >
        <Navbar />
      </motion.nav>

      {/* Main Content */}
      <motion.section
        id="hero"
        className="relative flex flex-col items-center justify-center py-16 z-10 text-white"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        <div className="container text-center p-10 mt-20">
          <motion.h1
            className="text-6xl font-bold mb-4"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            AI Doctor
          </motion.h1>
          <motion.h2
            className="text-3xl mb-6 font-bold"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
          >
            Simplified Healthcare Solutions
          </motion.h2>
        </div>

        <motion.form
          onSubmit={handleSubmit}
          className="space-y-6 p-6 rounded-lg shadow-lg bg-white bg-opacity-80 w-full max-w-md"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="form-group flex flex-col items-center">
            <label htmlFor="inputEmail4" className="text-lg font-semibold text-gray-700 mb-2">
              AI Doctor
            </label>
            <input
              type="text"
              className="w-full p-3 border rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
              id="inputEmail4"
              name="user_input"
              placeholder="Enter your Symptoms"
              value={formData.user_input}
              onChange={handleChange}
              required
            />
          </div>
          <button
            type="submit"
            className="w-full px-5 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition duration-300"
          >
            Make Predictions
          </button>
        </motion.form>

        <motion.button
          onClick={handlePopupOpen}
          className="mt-6 w-full max-w-md px-5 py-3 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition duration-300 shadow-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
        >
          Open Report
        </motion.button>

        <Popup
          open={isPopupOpen}
          onClose={() => setIsPopupOpen(false)}
          contentStyle={{
            width: "80%",
            padding: "20px",
            maxHeight: "600px",
            overflowY: "auto",
            borderRadius: "8px",
            boxShadow: "0 4px 10px rgba(0,0,0,0.2)",
          }}
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-2xl font-bold mb-4 text-gray-700">Report</h1>
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-gray-800">Disease: {report.Disease}</h2>
              <textarea
                value={report.Information}
                className="w-full p-4 border rounded-lg resize-none text-gray-700 shadow-inner bg-gray-50"
                style={{ height: "150px" }}
                readOnly
              />
              <div>
                <h3 className="text-lg font-bold text-gray-800">Predicted Medicines</h3>
                {report.Medicine && (
                  <ul className="mt-2 grid grid-cols-1 gap-2">
                    <li className="p-3 border rounded-lg shadow-sm bg-green-50 text-gray-800">{report.Medicine.M1}</li>
                    <li className="p-3 border rounded-lg shadow-sm bg-green-50 text-gray-800">{report.Medicine.M2}</li>
                    <li className="p-3 border rounded-lg shadow-sm bg-green-50 text-gray-800">{report.Medicine.M3}</li>
                  </ul>
                )}
              </div>
              <div>
                <h3 className="text-lg font-bold text-gray-800">Predicted Tests</h3>
                {report.Test && (
                  <ul className="mt-2 grid grid-cols-1 gap-2">
                    <li className="p-3 border rounded-lg shadow-sm bg-blue-50 text-gray-800">{report.Test.T1}</li>
                    <li className="p-3 border rounded-lg shadow-sm bg-blue-50 text-gray-800">{report.Test.T2}</li>
                    <li className="p-3 border rounded-lg shadow-sm bg-blue-50 text-gray-800">{report.Test.T3}</li>
                  </ul>
                )}
              </div>
            </div>
          </motion.div>
        </Popup>
      </motion.section>

      {/* Footer */}
      <motion.footer
        className="relative z-10"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
      >
        <Footer />
      </motion.footer>

      <ToastContainer />
    </div>
  );
};

export default ChatPage;
