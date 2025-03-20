
import React, { useState } from "react";
import Button from "@mui/material/Button";
import SendIcon from "@mui/icons-material/Send";
import emailjs from "emailjs-com";

function ContactForm() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState("");

  const handleSendEmail = (e) => {
    e.preventDefault();

    if (!name || !email || !message) {
      setStatus("Please fill in all fields.");
      return;
    }

    const templateParams = {
      from_name: name, // match this with the variable name in your EmailJS template
      from_email: email, // match this with the variable name in your EmailJS template
      message: message,
    };

    emailjs
      .send(
        "service_7au8s79", // Service ID
        "template_jj1ijmu", // Template ID
        templateParams,
        "B_mF1jpXMI8OC1BHa" // User ID
      )
      .then(
        (response) => {
          console.log("SUCCESS!", response.status, response.text);
          setStatus("Message sent successfully!");
          setName("");
          setEmail("");
          setMessage("");
        },
        (err) => {
          console.error("FAILED...", err);
          setStatus("Failed to send the message. Please try again.");
        }
      );
  };

  return (
    <div className="w-full  p-8 bg-transparent rounded-lg">
      <h1 className="text-4xl font-bold mb-4 text-center">Contact US</h1>
      <div className="h-4 bg-blue-400 mb-8 mx-auto rounded-full"></div>

      <form className="space-y-6" onSubmit={handleSendEmail}>
        <div>
          <label className="block text-sm font-medium" htmlFor="name">
            Name
          </label>
          <input
            type="text"
            id="name"
            className="w-full px-4 py-2 mt-2 bg-gray-100 text-gray-700 border-4 rounded-full focus:outline-none focus:ring-2 focus:ring-cyan-500"
            placeholder="Your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium" htmlFor="email">
            Email
          </label>
          <input
            type="email"
            id="email"
            className="w-full px-4 py-2 mt-2 bg-gray-100 text-gray-700 border-4  rounded-full focus:outline-none focus:ring-2 focus:ring-cyan-500"
            placeholder="Your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium" htmlFor="message">
            Message
          </label>
          <textarea
            id="message"
            className="w-full px-4 py-2 mt-2 bg-gray-50 text-gray-700 border-4 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-cyan-500"
            rows="4"
            placeholder="Message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            required
          ></textarea>
        </div>

        
          <Button type="submit"  variant="contained"  endIcon={<SendIcon />}>
            Send Message
          </Button>
        

        {status && (
          <p className="text-center mt-4 text-sm text-gray-300">{status}</p>
        )}
      </form>
    </div>
  );
}

export default ContactForm;