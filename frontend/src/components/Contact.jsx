
import { Mail, Phone, MapPin } from "lucide-react";
import { useState } from "react";
import axios from "axios";

const Contact = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });

  const [success, setSuccess] = useState(false);
  const [error, setError] = useState(false);

  // Handle input changes
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://127.0.0.1:8000/contacts/", formData);
      if (response.status === 201) {
        setSuccess(true);
        setError(false);
        setFormData({ name: "", email: "", message: "" });
      }
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
      setError(true);
      setSuccess(false);
    }
  };

  return (
    <div className="bg-gray-100 min-h-screen flex items-center justify-center px-6">
      <div className="bg-white shadow-lg rounded-lg p-8 max-w-4xl w-full">
        <h2 className="text-3xl font-bold text-center text-gray-800">Contact Us</h2>
        <p className="text-center text-gray-600 mt-2">
          Have questions? Send us a message, and we’ll get back to you!
        </p>

        <div className="grid md:grid-cols-2 gap-8 mt-8">
          {/* Contact Info */}
          <div className="space-y-6">
            <div className="flex items-center gap-3">
              <Mail className="text-blue-500" size={24} />
              <p className="text-gray-700">izzyy720@gmail.com</p>
            </div>
            <div className="flex items-center gap-3">
              <Phone className="text-blue-500" size={24} />
              <p className="text-gray-700">+250780463588</p>
            </div>
            <div className="flex items-center gap-3">
              <MapPin className="text-blue-500" size={24} />
              <p className="text-gray-700">Kigali, Rwanda</p>
            </div>
          </div>

          {/* Contact Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <input
              type="text"
              name="name"
              placeholder="Your Name"
              value={formData.name}
              onChange={handleChange}
              className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
              required
            />
            <input
              type="email"
              name="email"
              placeholder="Your Email"
              value={formData.email}
              onChange={handleChange}
              className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400"
              required
            />
            <textarea
              name="message"
              placeholder="Your Message"
              value={formData.message}
              onChange={handleChange}
              className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-400 h-32"
              required
            />
            <button
              type="submit"
              className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition"
            >
              Send Message
            </button>
          </form>
        </div>

        {/* Success/Error Messages */}
        {success && <p className="text-green-600 text-center mt-4">Message sent successfully!</p>}
        {error && <p className="text-red-600 text-center mt-4">Failed to send message!</p>}
      </div>
    </div>
  );
};

export default Contact;
