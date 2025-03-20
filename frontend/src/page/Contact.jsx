import { motion } from "framer-motion";
import ContactForm from "../components/ContactForm";
import Footer from "../components/Footer";
import Navbar from "../components/Navbar";

const Contact = () => {
  return (
    <div>
      {/* Animated Navbar */}
      <motion.div
        initial={{ opacity: 0, y: -100 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Navbar />
      </motion.div>

      {/* Animated Hero Section */}
      <motion.section
        id="hero"
        className="d-flex align-items-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        <div className="container mx-auto flex flex-col md:flex-row items-center justify-between py-12">
          <div className="w-full lg:w-[50%]">
            {/* Animated Text */}
            <motion.h2
              className="text-lg mt-10 text-gray-600"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              Need assistance or have questions? Contact AI Doctor via email, phone, or our online form for prompt support.
            </motion.h2>

            <div className="d-flex w-full ">
              <ContactForm />
            </div>
          </div>

          <div className=" flex w-full lg:w-[40%] mt-8 lg:mt-0 p-4">
            <motion.img
              src="/communications.png"
              className="img-fluid animate__animated animate__fadeIn"
              alt="Contact Illustration"
              initial={{ opacity: 0 , x: 150 }}
              animate={{ opacity: 1 , x:0}}
              transition={{ duration: 1 }}
            />
          </div>
        </div>
      </motion.section>

      {/* Animated Footer */}
      <motion.div
        initial={{ opacity: 0 , y: -50}}
        animate={{ opacity: 1 , y:0}}
        transition={{ duration: 0.8 }}
      >
        <Footer />
      </motion.div>
    </div>
  );
};

export default Contact;
