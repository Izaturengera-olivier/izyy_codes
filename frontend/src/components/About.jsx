import { Briefcase, Users, CheckCircle } from "lucide-react";
import image from "../assets/hello.jpg";

const About = () => {
  return (
    <div className="bg-gray-100 min-h-screen py-12 px-6">
      <div className="max-w-5xl mx-auto text-center">
        {/* Header Section */}
        <h1 className="text-4xl font-bold text-blue-600">About Us</h1>
        <p className="mt-4 text-lg text-gray-600">
          We are a passionate team dedicated to delivering the best services for our customers.
        </p>

        {/* Info Cards Section */}
        <div className="grid md:grid-cols-3 gap-8 mt-12">
          {/* Experience Card */}
          <div className="bg-white p-6 rounded-2xl shadow-lg hover:shadow-xl transition">
            <Briefcase size={50} className="text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold">10+ Years of Experience</h3>
            <p className="text-gray-600 mt-2">
              Our team brings years of expertise in delivering quality solutions.
            </p>
          </div>

          {/* Team Card */}
          <div className="bg-white p-6 rounded-2xl shadow-lg hover:shadow-xl transition">
            <Users size={50} className="text-green-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold">Dedicated Team</h3>
            <p className="text-gray-600 mt-2">
              We are a group of professionals working together to achieve excellence.
            </p>
          </div>

          {/* Success Card */}
          <div className="bg-white p-6 rounded-2xl shadow-lg hover:shadow-xl transition">
            <CheckCircle size={50} className="text-yellow-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold">100% Client Satisfaction</h3>
            <p className="text-gray-600 mt-2">
              We ensure the highest level of satisfaction for all our clients.
            </p>
          </div>
        </div>

        {/* Call to Action */}
        <div className="mt-12">
          <a
            href="/contact"
            className="bg-blue-600 text-white px-6 py-3 rounded-lg text-lg font-semibold hover:bg-blue-700 transition"
          >
            Get in Touch
          </a>
          <img src={image} alt="hello" className="w-48 h-48 mx-auto mt-6"/>
        </div>
      </div>
    </div>
  );
};

export default About;
