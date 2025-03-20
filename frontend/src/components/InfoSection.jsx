import React from "react";
import { motion } from "framer-motion";

const InfoSection = () => {
  const cards = [
    {
      id: "01",
      title: "AI-Powered Diagnosis",
      description:
        "AI Doctor leverages advanced machine learning algorithms to provide highly accurate and personalized diagnoses based on your symptoms.",
      extraInfo:
        "Our models are trained on diverse medical datasets to ensure reliability and inclusivity for users worldwide.",
    },
    {
      id: "02",
      title: "Medication Recommendations",
      description:
        "Receive tailored medication suggestions for your condition, ensuring the best possible care for your health.",
      extraInfo:
        "Our AI keeps you informed about potential side effects and alternative treatments for safe recovery.",
    },
    {
      id: "03",
      title: "Test Recommendations",
      description:
        "Get precise recommendations for lab tests based on your symptoms, enabling quicker and more accurate diagnoses.",
      extraInfo:
        "Save time and reduce costs with targeted testing guided by AI insights.",
    },
    {
      id: "04",
      title: "User-Friendly Interface",
      description:
        "Our platform is designed to be intuitive and easy to use, making healthcare accessible to users of all ages.",
      extraInfo:
        "With seamless navigation and clear instructions, AI Doctor ensures a hassle-free experience.",
    },
    {
      id: "05",
      title: "Global Healthcare Accessibility",
      description:
        "Breaking geographical barriers, AI Doctor offers healthcare solutions to users across the globe.",
      extraInfo:
        "No matter where you are, you can access expert medical advice at your fingertips.",
    },
  ];

  return (
    <motion.section
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      viewport={{ once: true }}
      className="mt-16 m-auto px-8 md:px-16 py-12 bg-gray-100 w-[95%] rounded-3xl"
    >
      <div className="space-y-16 ">
        {cards.map((card) => (
          <div key={card.id} className="grid md:grid-cols-2 gap-8 ">
            <div className="space-y-2">
              <h3 className="text-xl font-semibold">{card.id}</h3>
              <h2 className="text-2xl md:text-3xl font-bold leading-tight">
                {card.title}
              </h2>
            </div>
            <div>
              <p className="text-gray-700">{card.description}</p>
              <p className="mt-4 text-gray-600">{card.extraInfo}</p>
            </div>
          </div>
        ))}
      </div>
    </motion.section>
  );
};

export default InfoSection;
